import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete

class RobotSearchEnv(ParallelEnv):
    def __init__(self, num_robots=3, width=10, height=10, target_location=(8, 8), fov_radius=2):
        super().__init__()
        #parameters
        self.num_robots = num_robots
        self.width = width
        self.height = height
        self.target_location = target_location
        self.fov_radius = fov_radius

        # generating obstacles
        self.obstacle_coords = []
        self.generate_obstacles(4, 2, 2)

        # action and observation spaces
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(width, height, 1), dtype=np.float32)

        # randomize robot positions
        self.robot_positions = {f"robot_{i}": (np.random.randint(width), np.random.randint(height)) for i in range(num_robots)}

    def reset(self):
        # resets the environment to its initial state.
        self.robot_positions = {f"robot_{i}": (np.random.randint(self.width), np.random.randint(self.height)) for i in range(self.num_robots)}
        return self.get_observations()

    def step(self, actions):
        # executes action and updates environment
        rewards = {}
        done = {}
        info = {}

        for robot, action in actions.items():
            x, y = self.robot_positions[robot]
            dx, dy = action

            # move in bounds
            new_x = np.clip(x + dx, 0, self.width - 1)
            new_y = np.clip(y + dy, 0, self.height - 1)

            # check collisions
            if not self.is_collision(new_x, new_y):
                self.robot_positions[robot] = (new_x, new_y)

            # check for target
            if self.within_radius((new_x, new_y), self.target_location):
                rewards[robot] = 1.0
                done[robot] = True
            else:
                rewards[robot] = 0.0
                done[robot] = False

        # check if all robots are done
        done["__all__"] = any(done.values())

        return self.get_observations(), rewards, done, info

    def get_observations(self):
        #all observations for every robot
        observations = {}
        for robot, position in self.robot_positions.items():
            observations[robot] = self.make_observation(position)
        return observations

    def make_observation(self, position):
        # observation around a robot's position
        x, y = position
        obs = np.zeros((self.width, self.height, 1), dtype=np.float32)
        for i in range(self.width):
            for j in range(self.height):
                if self.within_radius((x, y), (i, j)) and not self.is_collision(i, j):
                    obs[i, j, 0] = 1.0
        return obs

    def within_radius(self, pos1, pos2):
        #for checking if target is within the radius of the robot
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) <= self.fov_radius

    def is_collision(self, x, y):
        # check if robot collides with obstacle
        for obs_x, obs_y, obs_w, obs_h in self.obstacle_coords:
            if obs_x <= x < obs_x + obs_w and obs_y <= y < obs_y + obs_h:
                return True
        return False

    def generate_obstacles(self, num_obstacles, obs_width, obs_height):
        # generates the obstacles
        obstacles = []
        for i in range(num_obstacles):
            while True:
                obs_x = np.random.randint(0, self.width - obs_width + 1)
                obs_y = np.random.randint(0, self.height - obs_height + 1)
                new_obstacle = (obs_x, obs_y, obs_width, obs_height)
                if not any(self.is_collision(x, y) for x in range(obs_x, obs_x + obs_width) for y in range(obs_y, obs_y + obs_height)):
                    self.obstacle_coords.append(new_obstacle)
                    break

if __name__ == "__main__":
    # initialize environment
    env = RobotSearchEnv()
    observations = env.reset()

    # run a quick random simulation
    num_steps = 10
    for step in range(num_steps):
        actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.robot_positions.keys()}
        observations, rewards, done, info = env.step(actions)

        print(f"Step {step + 1}:")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Done: {done}")

        if done["__all__"]:
            print("Simulation finished. Target found!")
            break