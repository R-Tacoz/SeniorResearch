import os
import numpy as np
import supersuit as ss
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import parallel_api_test
from utils.envs import MainEnv

# Testing environment
parallel_env = MainEnv(
    num_robots=3, 
    width=10, 
    height=10, 
    target_location=(8, 8), 
    camera_range=2, 
    render_mode="human"
    )
parallel_api_test(parallel_env)

# Wrap the environment for compatibility with Stable-Baselines3
env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, base_class="stable_baselines3")

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

# Define MAPPO model using PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=512,
    batch_size=64,
    n_epochs=10
)

reward_tracker = RewardTrackerCallback(check_freq=10000, verbose=1)
model.learn(total_timesteps=100000, callback=reward_tracker)

model.save("mappo_robot_search")

# Plot performance
plt.plot(reward_tracker.timesteps, reward_tracker.avg_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.title("Training Performance of MAPPO on Robot Search Task")
plt.show()

# Load and test the trained model
model = PPO.load("mappo_robot_search")
print(env.reset())
obs = env.reset()

for _ in range(50):
    actions, _ = model.predict(obs)
    obs, rewards, done, truncs = env.step(actions)
    env.render() 
    pygame.time.delay(100)

    if done[0] or done[1] or done[2]:
            break

