import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from torch import nn

from utils.envs import MainEnv, SpiralEnv, SquareHexEnv
from utils.agents import AgentCNN, AgentGNN
from utils.callbacks import TrainingMetricsCallback

#%% Hyperparameters
NUM_ENVS = 4  # Tune based on CPU
TOTAL_TIMESTEPS = 1_000_000
STEPS_PER_CHECKPOINT = 20_000

# LOAD_ROOT = "./saved_runs/run30"
# MODEL_LOAD_PATH = LOAD_ROOT + "/conv1mlp2-final.zip"
# ENV_LOAD_PATH = LOAD_ROOT + "/vecnormenv_state-final.pkl"
LOAD_ROOT = None
MODEL_LOAD_PATH = None
ENV_LOAD_PATH = None

SAVE_ROOT = "./saved_runs/rz2"
MODEL_NAME = "conv1mlp2" # name to save models with
MODEL_SAVE_PATH = SAVE_ROOT + "/checkpoints"
ENV_SAVE_PATH = SAVE_ROOT + "/vecnormenv_state-final.pkl" # won't work if you add directories between the file and the root
TENSORBOARD_LOG_PATH = SAVE_ROOT

def main():
    
#%% Environment
    
    env_width = 18
    env_height = 18
    
    env = MainEnv(
        max_episode_len=env_width*env_height*2,
        num_robots=3, 
        width=env_width, 
        height=env_height, 
        num_obstacles=10,
        obstacle_size=(1.5,1.5),
        render_mode="human"
    )
    # env = SpiralEnv(max_episode_len=env_width*env_height*4)
    # env = SquareHexEnv(
    #     width=env_width, 
    #     height=env_height, 
    #     square_length=None, 
    #     gap_size=None,
    # )
    
    comm_range = env.communication_range
    
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
    
#%% Agent Policy

    # policy_kwargs = dict(
    #     features_extractor_class=AgentCNN,
    #     features_extractor_kwargs=dict(
    #         features_dim=128,
    #     ),
    #     net_arch=[128,64],
    #     activation_fn=nn.ReLU,
    # ) 
    
    policy_kwargs = dict(
        features_extractor_class=AgentGNN,
        features_extractor_kwargs=dict(
            features_dim=128,
            n_conv_features = 32,
            message_size = 64,
            lidar_rays_count = 90,
            communication_range = comm_range,
        ),
        net_arch=[128,64],
        activation_fn=nn.ReLU,
    ) 

#%% Training
    
    lr_scheduler = get_linear_fn(start=1e-3, end=5e-5, end_fraction=0.7)
    
    train_hyperparams = dict(
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
    
    if LOAD_ROOT is not None:
        print("Loading model from", MODEL_LOAD_PATH)
        model = PPO.load(
            MODEL_LOAD_PATH, 
            **train_hyperparams,
        )
        new_logger = configure(TENSORBOARD_LOG_PATH, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
    else:
        model = PPO(
            policy="CnnPolicy",
            **train_hyperparams,
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

#%% Saving
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

#%% Main
if __name__=="__main__": main()
