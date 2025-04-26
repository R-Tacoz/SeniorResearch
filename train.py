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

# Callback to track average reward over training
class RewardTrackerCallback(BaseCallback):
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.avg_rewards = []
        self.timesteps = []

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            self.avg_rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
            
            if self.verbose:
                print(f"Step: {self.num_timesteps}, Avg Reward: {mean_reward:.4f}")
            
            self.episode_rewards = []
        return True
    
    def _on_rollout_end(self):
        # Store episode rewards after each rollout
        if "rewards" in self.locals:
            mean_reward = np.mean(self.locals["rewards"])
            self.episode_rewards.append(mean_reward)
        else:
            print("Warning: 'rewards' not found in self.locals")

SAVE_DIR = "saved_runs/run1"

def main():

    # RL Environment
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
    # parallel_api_test(parallel_env)

    # Wrap the pettingzoo environment for compatibility with Stable-Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")

    # Define MAPPO model using PPO
    model = PPO(
        "MlpPolicy", # see stable_baselines3.common.policies.ActorCriticPolicy
        env,
        # policy_kwargs={
        #     "net_arch": [32, 32]
        # },
        verbose=1,
        learning_rate=0.0001,
        gamma=0.99,
        n_steps=256,
        batch_size=32,
        n_epochs=10,
        device='cpu', # says gpu only for CNN policies
    )

    print(f"Setup complete. Trained model will be saved to ./{SAVE_DIR}")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    reward_tracker = RewardTrackerCallback(check_freq=100, verbose=1)
    model.learn(total_timesteps=256*150, callback=reward_tracker)

    print(f"Training complete. Saving...")
    model.save(SAVE_DIR)

    # Plot performance
    plt.plot(reward_tracker.timesteps, reward_tracker.avg_rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.title("Training Performance of MAPPO on Robot Search Task")
    plt.show()

if __name__=="__main__":
    main()