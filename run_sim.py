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

def main():
    load_dir = "saved_runs/run1"
    
    # Testing environment
    parallel_env = MainEnv(
        num_robots=3, 
        width=20, 
        height=20, 
        num_obstacles=10,
        target_location=None, 
        lidar_range=5,
        camera_range=8,
        success_range=1,
        render_mode="human"
        )

    # Wrap the environment for compatibility with Stable-Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")

    # Load and test the trained model
    model = PPO.load(load_dir, device='cpu')
    #print(env.reset())
    obs = env.reset()

    for _ in range(50000):
        actions, _ = model.predict(obs)
        obs, rewards, terms, truncs = env.step(actions)
        env.render() 
        pygame.time.delay(100)


if __name__=="__main__":
    main()