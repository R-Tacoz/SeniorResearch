import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from utils.envs import MainEnv

# from sb3_contrib.ppo_mask import MaskablePPO  # If using action masking

import os

# === Hyperparameters ===
NUM_ENVS = 4  # Tune based on CPU
TOTAL_TIMESTEPS = 4_000_000
STEPS_PER_CHECKPOINT = 1_000 * NUM_ENVS
SAVE_ROOT = "./saved_runs/run1"
SAVE_PATH = SAVE_ROOT + "/ppo_agent"
TENSORBOARD_LOG = SAVE_ROOT + "/ppo_run"


def main():
    
    
    env = MainEnv(
        num_robots=3, 
        width=18, 
        height=18, 
        num_obstacles=10,
        target_location=None, 
        lidar_range=5,
        camera_range=8,
        success_range=1,
        render_mode="human"
    )
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=NUM_ENVS, base_class="stable_baselines3")

    # Optional: Reward normalization (leave obs normalization off if done manually)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Optional: Track rewards and lengths per episode
    env = VecMonitor(env)

    # === Setup PPO Agent ===
    policy_kwargs = dict(net_arch=[128, 128, 64])  # Your architecture

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu" #"auto"
    )

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PER_CHECKPOINT,
        save_path=SAVE_PATH,
        name_prefix="ppo_model"
    )

    # === Train ===
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name=TENSORBOARD_LOG
    )

    print("Training complete. Saving...", end=" ")
    # === Save Final Model and Normalization Stats ===
    model.save(os.path.join(SAVE_PATH, "final_model"))
    
    if isinstance(env.unwrapped, VecNormalize):
        env.get_attr("venv")[0].save(os.path.join(SAVE_ROOT, "/envs/vecnorm1.pkl"))
    elif isinstance(env.unwrapped, VecNormalize):
        env.unwrapped.save(os.path.join(SAVE_ROOT, "/envs/vecnorm1.pkl"))

    # env.save(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
    
    print("Done.")


if __name__=="__main__": main()