import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_robot as wr
import numpy as np

register(
    id='robot-v0',                               
    entry_point='v0_robot_env:RobotEnv',
)

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, grid_rows=63, grid_cols=63, render_mode=None):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        self.robot = wr.Robot(grid_rows=grid_rows, grid_cols=grid_cols, fps=self.metadata['render_fps'])

        self.action_space = spaces.Discrete(len(wr.RobotAction))

        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]),
            shape=(4,),
            dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.reset(seed=seed)

        obs = np.concatenate((self.robot.robot_pos, self.robot.target_pos))
        
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, info

    def step(self, action):
        target_reached = self.robot.perform_action(wr.RobotAction(action))

        reward=0
        terminated=False
        if target_reached:
            reward=1
            terminated=True

        obs = np.concatenate((self.robot.robot_pos, self.robot.target_pos))

        info = {}

        if(self.render_mode=='human'):
            print(wr.RobotAction(action))
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        self.robot.render()

if __name__=="__main__":
    env = gym.make('robot-v0', render_mode='human')

    obs = env.reset()[0]

    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]