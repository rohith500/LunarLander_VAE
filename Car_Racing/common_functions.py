import cv2
import numpy as np

def process_state_image(frame):
    """Process image frames: Resize to 96x96."""
    obs = cv2.resize(frame, (96, 96))  
    return np.round(obs).astype(np.uint8)  

def generate_state_frame_stack_from_queue(deque):
    """Stack frames for the DQN agent."""
    frame_stack = np.array(deque)  
    if frame_stack.shape == (3, 96, 96, 3):  
        return np.mean(frame_stack, axis=0)  
    else:
        raise ValueError(f"Unexpected shape in frame stack: {frame_stack.shape}")
