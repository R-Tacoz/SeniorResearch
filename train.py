import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from torch import nn
from utils.envs import MainEnv, SpiralEnv, SquareHexEnv
from utils.agents import AgentCNN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os

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

# from sb3_contrib.ppo_mask import MaskablePPO  # If using action masking

# === Hyperparameters ===
NUM_ENVS = 4  # Tune based on CPU
TOTAL_TIMESTEPS = 200_000
STEPS_PER_CHECKPOINT = 10_000

LOAD_ROOT = "./saved_runs/run30"
MODEL_LOAD_PATH = LOAD_ROOT + "/conv1mlp2-final.zip"
ENV_LOAD_PATH = LOAD_ROOT + "/vecnormenv_state-final.pkl"
# LOAD_ROOT = None
# MODEL_LOAD_PATH = None
# ENV_LOAD_PATH = None

SAVE_ROOT = "./saved_runs/run30"
MODEL_NAME = "conv1mlp2" # name to save models with
MODEL_SAVE_PATH = SAVE_ROOT + "/checkpoints"
ENV_SAVE_PATH = SAVE_ROOT + "/vecnormenv_state-final.pkl" # won't work if you add directories between the file and the root
TENSORBOARD_LOG_PATH = SAVE_ROOT


def main():
    
    env_width = 15
    env_height = 15
    
    env = MainEnv(
        max_episode_len=env_width*env_height*2,
        num_robots=2, 
        width=env_width, 
        height=env_height, 
        num_obstacles=30,
        target_location=None, 
        lidar_range=3,
        camera_range=2,
        success_range=1,
        render_mode="human"
    )
    # env = SpiralEnv(max_episode_len=env_width*env_height*4)
    # env = SquareHexEnv(
    #     width=env_width, 
    #     height=env_height, 
    #     square_length=None, 
    #     gap_size=None,
    # )
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=NUM_ENVS, base_class="stable_baselines3")

    # Optional: Reward normalization (leave obs normalization off if done manually)
    if LOAD_ROOT is not None:
        try:
            env = VecNormalize.load(ENV_LOAD_PATH, env)
            print("Loaded VecNormalize from", ENV_LOAD_PATH)
        except FileNotFoundError:
            print("WARNING: Attempted to load VecNormalize, but not file was found. Using new VecNormalize.")
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Optional: Track rewards and lengths per episode
    env = VecMonitor(env)

    # === Setup PPO Agent ===
    policy_kwargs = dict(
        features_extractor_class=AgentCNN,
        features_extractor_kwargs=dict(
            features_dim=128,
        ),
        net_arch=[128,64],#[256, 256, 256, 128],
        activation_fn=nn.ReLU,
    ) 
    
    lr_scheduler = get_linear_fn(start=1e-3, end=5e-5, end_fraction=0.7)
    
    if LOAD_ROOT is not None:
        print("Loading model from", MODEL_LOAD_PATH)
        model = PPO.load(
            MODEL_LOAD_PATH, 
            env=env, 
            learning_rate=lr_scheduler,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.992,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.55,
            # ent_coef=0.03,
            max_grad_norm=0.5,
            tensorboard_log=TENSORBOARD_LOG_PATH,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto"
        )
        new_logger = configure(TENSORBOARD_LOG_PATH, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=lr_scheduler,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.992,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.55,
            ent_coef=0.03,
            max_grad_norm=0.5,
            tensorboard_log=TENSORBOARD_LOG_PATH,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto"
        )

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PER_CHECKPOINT,
        save_path=MODEL_SAVE_PATH,
        name_prefix=MODEL_NAME,
        save_vecnormalize=True,
    )

    metrics_callback = TrainingMetricsCallback()

    callback_list = CallbackList([metrics_callback, checkpoint_callback])

    # === Train ===
    model.num_timesteps = 0
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        progress_bar=True,
        tb_log_name="tb_log"
    )

    print("Training complete. Saving...")

    metrics_callback.plot_metrics()

    # === Save Final Model and Normalization Stats ===
    model.save(os.path.join(SAVE_ROOT, MODEL_NAME + "-final"))
    
    
    if isinstance(env, VecNormalize):
        env.save(ENV_SAVE_PATH)
    else:
        try:
            env.venv.save(ENV_SAVE_PATH)
        except:
            print("Unable to locate VecNormalize in wrapper stack. No env saving will occur")
    
    print("Done.")


if __name__=="__main__": main()
