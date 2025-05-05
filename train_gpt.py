import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from utils.envs import MainEnv

# from sb3_contrib.ppo_mask import MaskablePPO  # If using action masking

import os

# === Hyperparameters ===
NUM_ENVS = 4  # Tune based on CPU
TOTAL_TIMESTEPS = 500_000
STEPS_PER_CHECKPOINT = 1_000 * NUM_ENVS

LOAD_ROOT = "./saved_runs/run2-0"
MODEL_LOAD_PATH = LOAD_ROOT + "/ppo_agent/_final-mlp3.zip"
ENV_LOAD_PATH = LOAD_ROOT + "/envs/vecnormenv_state.pkl"
# LOAD_ROOT = None
# MODEL_LOAD_PATH = None
# ENV_LOAD_PATH = None

SAVE_ROOT = "./saved_runs/run2-1"
MODEL_NAME = "mlp3" # name to save models with
MODEL_SAVE_PATH = SAVE_ROOT + "/ppo_agent"
ENV_SAVE_PATH = SAVE_ROOT + "/_final-vecnormenv_state.pkl" # won't work if you add directories between the file and the root
TENSORBOARD_LOG_PATH = SAVE_ROOT + "/tb_logs"


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
    if LOAD_ROOT is not None:
        try:
            env = VecNormalize.load(ENV_LOAD_PATH, env)
            print("Loaded VecNormalize from", ENV_LOAD_PATH)
        except FileNotFoundError:
            print("WARNING: Attempted to load VecNormalize, but not file was found. Using new VecNormalize.")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Optional: Track rewards and lengths per episode
    env = VecMonitor(env)

    # === Setup PPO Agent ===
    policy_kwargs = dict(net_arch=[128, 128, 64])  # Your architecture
    
    lr_scheduler = get_linear_fn(start=5e-4, end=8e-5, end_fraction=0.8)

    if LOAD_ROOT is not None:
        print("Loading model from", MODEL_LOAD_PATH)
        model = PPO.load(MODEL_LOAD_PATH, env, device='cpu')
        new_logger = configure(TENSORBOARD_LOG_PATH, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr_scheduler,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=TENSORBOARD_LOG_PATH,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu" #"auto"
        )

    # === Callbacks ===
    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PER_CHECKPOINT,
        save_path=MODEL_SAVE_PATH,
        name_prefix=MODEL_NAME,
        save_vecnormalize=True,
    )

    # === Train ===
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        # tb_log_name="tb_log"
    )

    print("Training complete. Saving...")
    # === Save Final Model and Normalization Stats ===
    model.save(os.path.join(MODEL_SAVE_PATH, "_final-" + MODEL_NAME))
    
    
    if isinstance(env, VecNormalize):
        env.save(ENV_SAVE_PATH)
    else:
        try:
            env.venv.save(ENV_SAVE_PATH)
        except:
            print("Unable to locate VecNormalize in wrapper stack. No env saving will occur")

    # env.save(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
    
    print("Done.")


if __name__=="__main__": main()