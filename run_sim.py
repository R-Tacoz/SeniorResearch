import os
import numpy as np
import supersuit as ss
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import parallel_api_test
from utils.envs import MainEnv
from pynput import keyboard

FRAMERATE = 16 # also equals tickrate
SIM_LENGTH = 100000 # frames/ticks
MAX_VELO = 4 # cells / s
MAX_DISP = 2
N_ROBOTS = 3

end_sim = False
reset_sim = False

def on_press(key):
    global end_sim, reset_sim
    if isinstance(key, keyboard.KeyCode):
        if key.char == 'q':
            end_sim = True    
        elif key.char == 'r':
            reset_sim = True
        
def main():
    global reset_sim
    listener = keyboard.Listener(on_press)
    listener.start()
    
    load_dir = "saved_runs/run6/ppo_agent/ppo_model_1440000_steps"
    
    # Testing environment
    parallel_env = MainEnv(
        num_robots=N_ROBOTS, 
        width=18, 
        height=18, 
        num_obstacles=10,
        target_location=None, 
        lidar_range=5,
        camera_range=8,
        success_range=1,
        framerate=FRAMERATE,
        render_mode="human",
        )

    # Wrap the environment for compatibility with Stable-Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")

    # Load and test the trained model
    model = PPO.load(load_dir, device='cpu')
    #print(env.reset())
    obs = env.reset()

    for _ in range(5000):
        actions, _ = model.predict(obs)
        obs, rewards, terms, truncs = env.step(actions)
        
        # Update the original environment with the same actions
        # Convert actions from the wrapped format back to the PettingZoo format
        agent_actions = {}
        for i, agent in enumerate(parallel_env.agents):
            # Extract the appropriate action for this agent from the flattened array
            agent_actions[agent] = actions[i]
        
        # Step the original environment with the same actions
        _, _, terms, _, _ = parallel_env.step(agent_actions)
        
        # Render the original environment
        parallel_env.render()
        
        # Use a small delay to make the rendering visible
        pygame.time.delay(100)
        
        # Check if the original environment is still active
        if end_sim: #not parallel_env.active:# or any(terms.values()):
            break
        elif reset_sim:
            obs = env.reset()
            parallel_env.reset()
            reset_sim = False
            
    env.close()


if __name__=="__main__":
    main()