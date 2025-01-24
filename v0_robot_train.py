import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
import os
import v0_robot_env

def run_q(episodes, is_training=True, render=False):

    env = gym.make('robot-v0', render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.unwrapped.grid_rows, env.unwrapped.grid_cols, env.action_space.n))
    else:
        f = open('v0_solution.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9   
    discount_factor_g = 0.9 
    epsilon = 1             

    steps_per_episode = np.zeros(episodes)

    step_count=0
    for i in range(episodes):
        if(render):
            print(f'Episode {i}')

        state = env.reset()[0]
        terminated = False

        while(not terminated):

            if is_training and random.random() < epsilon:
                action = env.action_space.sample()
            else:                
                q_state_idx = tuple(state) 
                # print(f"State: {state}, Grid Rows: {env.unwrapped.grid_rows}, Grid Cols: {env.unwrapped.grid_cols}")
                # print(f"Q-table shape: {q.shape}")
                action = np.argmax(q[q_state_idx])
            
            new_state,reward,terminated,_,_ = env.step(action)

            q_state_action_idx = tuple(state) + (action,)

            q_new_state_idx = tuple(new_state)

            if is_training:
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            state = new_state

            step_count+=1
            if terminated:
                steps_per_episode[i] = step_count
                step_count = 0

        epsilon = max(epsilon - 1/episodes, 0)

    env.close()

    sum_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(steps_per_episode[max(0, t-100):(t+1)]) # Average steps per 100 episodes
    plt.plot(sum_steps)
    plt.savefig('v0_solution.png')

    if is_training:
        f = open("v0_solution.pkl","wb")
        pickle.dump(q, f)
        f.close()

# def train_sb3():
#     model_dir = "models"
#     log_dir = "logs"
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)

#     env = gym.make('warehouse-robot-v0')

#     model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
#     TIMESTEPS = 1000
#     iters = 0
#     while True:
#         iters += 1

#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#         model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}")

# def test_sb3(render=True):

#     env = gym.make('warehouse-robot-v0', render_mode='human' if render else None)

#     model = A2C.load('models/a2c_2000', env=env)

#     obs = env.reset()[0]
#     terminated = False
#     while True:
#         action, _ = model.predict(observation=obs, deterministic=True)
#         obs, _, terminated, _, _ = env.step(action)

#         if terminated:
#             break

if __name__ == '__main__':

    # Train/test using Q-Learning
    run_q(10, is_training=True, render=False)
    run_q(3, is_training=False, render=True)