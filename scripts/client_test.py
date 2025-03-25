from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8888)

num_steps=100
img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
wrist_img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
joint_state = np.random.uniform(-1, 1, 12)
gripper_state = np.random.choice([0, 1])
state = np.concatenate((joint_state, [gripper_state]))
task_instruction = "set_table: pick_013_apple"
for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    
    observation = {
        "image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    print("Step:", step)  
    print("Action chunk shape:", action_chunk.shape)
    print("Action chunk:", action_chunk[0])
 
