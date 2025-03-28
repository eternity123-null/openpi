"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
from pathlib import Path

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_ids is None or len(data_config.repo_ids) == 0:
        raise ValueError("Data config must have at least one repo_id")
    
    # Create a combined dataset from all repo_ids
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)
    
    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing combined stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Save combined stats to a single file (use the first repo_id or a combined name)
    if len(data_config.repo_ids) == 1:
        output_dir = config.assets_dirs / data_config.repo_ids[0]
    else:
        # 生成合并目录名，例如: repo1+repo2+repo3
        merged_name = "+".join(sorted(repo_id.split("/")[-1] for repo_id in data_config.repo_ids))
        output_dir = config.assets_dirs / merged_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing combined stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
