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
import torch
from torch import nn
from utils.envs import MainEnv
from utils.agents import AgentCNN

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
            
class EnvGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # get global graph embedding
        global_emb = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return global_emb

from stable_baselines3.common.policies import ActorCriticPolicy

class GNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # local observation encoder
        self.local_obs_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # GNN module
        self.env_gnn = EnvGNN(in_channels=4, hidden_channels=32, out_channels=32)

        # final policy + value heads
        combined_input_size = 32 + 32
        self.policy_net = nn.Linear(combined_input_size, self.action_dist.proba_distribution.mean_actions.shape[0])
        self.value_net = nn.Linear(combined_input_size, 1)

    def forward(self, obs, graph_data):
        local_feat = self.local_obs_net(obs)
        global_feat = self.env_gnn(graph_data)

        combined = torch.cat([local_feat, global_feat], dim=-1)
        return combined

    def _predict(self, obs, graph_data):
        combined = self.forward(obs, graph_data)
        actions = self.policy_net(combined)
        values = self.value_net(combined)
        return actions, values
        
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
        policy_kwargs=dict(
            net_arch = [64, 64, 32],
            activation_fn = nn.ReLU
        ),
        verbose=1,
        learning_rate=0.0001,
        gamma=0.99,
        n_steps=256,
        batch_size=32,
        n_epochs=10,
        device='cpu', # says gpu only for CNN policies
    )
    
    # model = PPO(
    #     "CnnPolicy", # see stable_baselines3.common.policies.ActorCriticPolicy
    #     env,
    #     policy_kwargs=dict(
    #         features_extractor_class=AgentCNN,
    #         features_extractor_kwargs=dict(
    #             features_dim=128
    #         ),
    #         net_arch = dict(
    #             pi=[128, 128, 64],
    #             vf=[128, 128, 64],
    #         ),
    #         activation_fn = nn.ReLU,
            
    #     ),
    #     verbose=1,
    #     learning_rate=0.0001,
    #     gamma=0.99,
    #     n_steps=256,
    #     batch_size=32,
    #     n_epochs=10,
    #     device='cpu', # says gpu only for CNN policies
    # )

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
