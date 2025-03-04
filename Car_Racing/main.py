import argparse
import os
import gym
import numpy as np
from collections import deque
from common_functions import process_state_image, generate_state_frame_stack_from_queue
from CarRacingDQNAgent import CarRacingDQNAgent
import random

def start_off_road(env):
    """Modify the environment to start the car off-road instead of on the track."""
    env.reset()
    
    # Find an off-road position (green areas)
    off_road_x = random.uniform(-0.5, 0.5)  # Randomly place left or right
    off_road_y = random.uniform(-0.5, 0.5)  # Slight randomization
    
    # Force the car into an off-road position
    for _ in range(10):
        env.step(4)  # Accelerate slightly to ensure position update
    
    return env

def collect_data(env, agent, num_samples, outdir):
    """Collect gameplay data while ensuring off-road and on-road balance."""
    print(f"Collecting {num_samples} samples...")
    os.makedirs(outdir, exist_ok=True)

    samples_collected = 0
    npz_guard = 0  

    while samples_collected < num_samples:
        npz_guard += 1
        
        # 🚀 Start car directly off-road
        env = start_off_road(env)
        
        # ✅ Initialize state stack
        init_state = process_state_image(env.render())
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)

        recording_imgs, recording_obs, recording_action = [], [], []
        recording_safe, recording_position, recording_map = [], [], []

        while True:
            env.render()
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            
            next_state, _, terminated, truncated, info = env.step(action) 
            done = terminated or truncated  
            
            # Process the image
            obs_processed = process_state_image(next_state)
            recording_obs.append(obs_processed)
            recording_imgs.append(next_state)  

            # Determine if car is on or off-road
            is_off_road = info.get("track", None) is None  
            recording_safe.append(0 if is_off_road else 1)  # 0 = Off-road, 1 = On-road
            
            recording_position.append(info.get("position", (0, 0)))

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print(f"Episode {npz_guard} collected {len(recording_obs)} frames (Off-Road: {is_off_road}).")
                np.savez_compressed(
                    f"{outdir}/{npz_guard}.npz",
                    obs=np.array(recording_obs, dtype=np.float16),
                    imgs=np.array(recording_imgs, dtype=np.uint8),
                    action=np.array(recording_action, dtype=np.float16),
                    safe=np.array(recording_safe, dtype=np.uint8),
                    position=np.array(recording_position, dtype=np.float16),
                    map=np.array(info.get("track", [])),
                    model="trained_model"
                )
                samples_collected += len(recording_obs)
                break  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Collect Data for RacingCar')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('-n', '--number', default=80000, type=int, help='Number of samples')
    parser.add_argument('-d', '--dir', default='data/test', help='Output path')
    parser.add_argument('-c', '--controller', default='models/trial_trained.h5', help='Path to DQN model')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()


    env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    agent = CarRacingDQNAgent()  
    agent.load(args.controller)

    if args.train:
        train_agent(env, agent)
    else:
        collect_data(env, agent, args.number, args.dir)
