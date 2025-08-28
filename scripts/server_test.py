from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

def make_autolife_example() -> dict:
    """Creates a random input example for the Autolife policy."""
    return {
        "state": np.random.rand(20),
        "eef_pos": np.random.rand(6),
        "head_left_image": np.random.randint(256, size=(3, 512, 512), dtype=np.uint8),
        "hand_left_image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "hand_right_image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "task": "do something",
    }

while True:
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = make_autolife_example()

    


    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    print("Action chunk: ",action_chunk,"\n\n")
    # Execute the actions in the environment.
    
