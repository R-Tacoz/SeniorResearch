import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import matplotlib.pyplot as plt

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []

        self.reward_window = deque(maxlen=100)
        self.success_window = deque(maxlen=100)
        self.steps_window = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Access info dict returned by the env's step function
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for info, done, reward in zip(infos, dones, rewards):
            if done:
                episode_reward = info.get("episode_reward", 0)
                success = info.get("success", 0)
                episode_length = info.get("episode_length", 0)

                self.episode_rewards.append(episode_reward)
                self.episode_successes.append(success)
                self.episode_lengths.append(episode_length)

                self.reward_window.append(episode_reward)
                self.success_window.append(success)
                self.steps_window.append(episode_length)

        return True

    def plot_metrics(self):
        print("Plotting Metrics...\r")
        
        episodes = range(len(self.episode_rewards))
        window_size = 100

        plt.figure(figsize=(18,5))

        # Episode Reward
        plt.subplot(1,3,1)
        plt.plot(episodes, self.episode_rewards, label="Reward")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # Success Rate (moving average)
        if len(self.episode_successes) >= window_size:
            success_rate_ma = np.convolve(self.episode_successes, np.ones(window_size)/window_size, mode='valid')
            plt.subplot(1,3,2)
            plt.plot(success_rate_ma, label="Success Rate (MA)")
            plt.title("Success Rate")
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.grid(True)

        # Steps to Target (moving average)
        if len(self.episode_lengths) >= window_size:
            steps_ma = np.convolve(self.episode_lengths, np.ones(window_size)/window_size, mode='valid')
            plt.subplot(1,3,3)
            plt.plot(steps_ma, label="Avg Steps (MA)")
            plt.title("Steps to Target")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.grid(True)

        plt.tight_layout()
        plt.show()
