"""
Script for converting a ManiSkill dataset to LeRobot format.

Usage:
uv run convert_maniskill_to_lerobot.py --dataset_file /path/to/your/data.h5

If you want to push your dataset to the Hugging Face Hub, you can use:
uv run convert_maniskill_to_lerobot.py --dataset_file /path/to/your/data.h5 --push_to_hub

Author: Chendong Zeng
Date:   2025/03/21

"""
import os
os.environ["LEROBOT_HOME"] = "/data/vla_dataset/mani_lerobot"
import shutil
import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
import tyro
from pathlib import Path
import re
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json  # 用于加载 JSON 文件


def load_h5_data(data):
    """
    Recursively load all HDF5 datasets into memory.
    """
    out = {}
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def main(dataset_file: str, *, push_to_hub: bool = False, load_count: int = -1):
    """
    将 ManiSkill 数据集转换为 LeRobot 格式并保存到 $LEROBOT_HOME 下。

    参数:
      - dataset_file: ManiSkill 数据集的 .h5 文件路径。
      - push_to_hub: 是否将转换后的数据集推送到 Hugging Face Hub。
      - load_count: 加载的轨迹数量，-1 表示加载所有轨迹。
    """
    # 设置输出仓库名称（同时也作为 Hugging Face Hub 上的 repo_id）
    REPO_NAME = "zcd/mani_lerobot"  # 请替换为你的 Hugging Face 用户名及仓库名称

    # 清理已存在的输出目录
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建 LeRobot 数据集，定义各项特征。注意这里的 shape 需与你数据的实际形状匹配
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="fetch",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (13,),  # 根据实际情况调整状态向量的维度
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (13,),  # 根据实际情况调整动作向量的维度
                "names": ["actions"],
            },
            "state_v": {
                "dtype": "float32",
                "shape": (12,),  # 根据实际情况调整状态向量的维度
                "names": ["state_v"],
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 打开 ManiSkill 数据集的 h5 文件和对应的 JSON 文件
    data = h5py.File(dataset_file, "r")
    path_obj = Path(dataset_file)
    task_name = path_obj.parent.parent.name  # set_table
    action_name = path_obj.parent.name  # pick
    # 获取文件名（不含扩展名），这里为 "013_apple"
    file_stem = path_obj.stem
    # 使用正则表达式移除文件名前缀的数字及下划线，得到 "apple"
    obj_name = re.sub(r"^\d+_", "", file_stem)
    task_desc = f"{task_name}: {action_name}_{obj_name}" # set_table: pick_apple
    
    json_file = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_file)
    episodes = json_data["episodes"]
    

    if load_count == -1:
        load_count = len(episodes) # load_count = 200
    print(f"Loading {load_count} episodes from {dataset_file}")
    # 遍历每个轨迹（episode）
    for eps_id in range(load_count):
        eps = episodes[eps_id]
        traj_key = f"traj_{eps['episode_id']}"
        if traj_key not in data:
            continue

        # 加载轨迹数据（转换为内存中的 numpy 数组） 数据已经都是ndarray
        trajectory = data[traj_key]
        trajectory = load_h5_data(trajectory)
        actions = np.array(trajectory["actions"], dtype=np.float32) 
        # actions_dim: [T, 12+1] (joint+gripper) control_mode: pd_joint_delta_pos
        print("actions_dim:",actions.shape)
        eps_len = len(actions)
        print(f"Processing episode {eps_id} with {eps_len} frames")
        # 假设所有观察数据均存储在 "obs" 下，并包含所需的键
        # obs: [T,Dict]
        obs = trajectory["obs"]
        # print("obs_keys:",obs)
        qposs = np.array(obs["agent"]["qpos"],dtype=np.float32) # [T+1, 12]
        qvels = np.array(obs["agent"]["qvel"],dtype=np.float32) # [T+1, 12]
        gripper = np.array(obs["extra"]["is_grasped"],dtype=np.float32) # [T+1]
        gripper = gripper.reshape(-1,1)
        print("gripper_dim:",gripper.shape)
        states = np.concatenate((qposs,gripper), axis=1)
        print("states_dim:",states.shape)
        images = obs["sensor_data"]["fetch_head"]["rgb"] # img_dim: (T+1, 128, 128, 3)
        # print("img_type:",type(images))
        wrist_images = obs["sensor_data"]["fetch_hand"]["rgb"]
        img_dim = (len(wrist_images),len(wrist_images[0]),len(wrist_images[0][0]),len(wrist_images[0][0][0]))
        print("img_dim:",img_dim)
        
        # exit()
        # 遍历轨迹中的每一帧
        for i in range(eps_len):
            if "fetch_head" in obs["sensor_data"]:
                tmp_image = images[i]
            else:
                tmp_image = np.zeros((128, 128, 3), dtype=np.uint8)
                
            if "fetch_hand" in obs["sensor_data"]:
                tmp_wrist_image = wrist_images[i]
            else:
                tmp_wrist_image = np.zeros((128, 128, 3), dtype=np.uint8)
            

            if "agent" in obs:
                tmp_state = states[i] 
                # print("state:",tmp_state)
                tmp_state_v = qvels[i]
            else:
                tmp_state = np.zeros(13, dtype=np.float32)
                tmp_state_v = np.zeros(12, dtype=np.float32)

            # 获取动作数据
            tmp_action = actions[i]
            # print("tmp_action:",tmp_action)
            if tmp_action.shape != (13,):
                print(tmp_action.shape)
                raise ValueError(f"Expected actions shape (13,), got {actions.shape}")
            # print("type of tmp_state:", type(tmp_state))
            # print("type of actions:", type(actions))


            # 将当前帧数据添加到 LeRobot 数据集中
            dataset.add_frame({
                "image": tmp_image,
                "wrist_image": tmp_wrist_image,
                "state": tmp_state,
                "actions": tmp_action,
                "state_v": tmp_state_v
            })

        dataset.save_episode(task=task_desc)

    # 整理数据集（合并所有帧并生成索引，统计信息可后续再计算）
    dataset.consolidate(run_compute_stats=False)

    # 可选：推送数据集到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["maniskill", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
