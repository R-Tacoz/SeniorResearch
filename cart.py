import gymnasium as gym

from stable_baselines3 import A2C

model = A2C("MlpPolicy", 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tb")
model.learn(total_timesteps=10000)