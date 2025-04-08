import numpy as np
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
from scipy.spatial import distance
from scipy.spatial import KDTree

def env(**kwargs):
    env_ = parallel_env(**kwargs)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env_)
    env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")
    return env_

class MultiAgentEnv:
    def __init__(self):
        self.visited_points = []

    def mark_visited(self, agent_x, agent_y):
        self.visited_points.append((agent_x, agent_y))

    def distance_to_nearest_explored(self, x, y):
        if not self.visited_points:
            return 0  # No points visited yet
        distances = [distance.euclidean((x, y), point) for point in self.visited_points]
        return min(distances)

    def calculate_reward(self, agent_x, agent_y):
        dist = self.distance_to_nearest_explored(agent_x, agent_y)
        return dist * 0.1

class RobotSearchEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "robot_search_v0"} 

    def __init__(self, num_robots=3, width=10, height=10, target_location=(8, 8), fov_radius=2, render_mode = None, seed = None, options = None):
        super().__init__()
        self.multiagent_env = MultiAgentEnv()
        #parameters
        self.render_mode = render_mode
        self.num_robots = num_robots
        self.width = width
        self.height = height
        self.target_location = target_location
        self.fov_radius = fov_radius
        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.agents = self.possible_agents[:]
        self.timestep = 0

        # generating obstacles
        self.obstacle_coords = []
        self.generate_obstacles(4, 2, 2)

        # action and observation spaces
        # self.action_spaces = {agent: Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in self.possible_agents}
        # self.observation_spaces = {agent: Box(low=0, high=1, shape=(width, height, 1), dtype=np.float32) for agent in self.possible_agents}
        # randomize robot positions
        self.robot_positions = {f"robot_{i}": (np.random.randint(width), np.random.randint(height)) for i in range(num_robots)}

        #pygame render initialization
        self.cell_size = 50 
        self.window_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.screen = None  
        self.clock = None  

    def reset(self, seed=None, options = None):
        # resets the environment to its initial state.
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        self.multiagent_env = MultiAgentEnv()

        self.obstacle_coords = []
        self.generate_obstacles(num_obstacles=4, obs_width=2, obs_height=2)

        while True:
            self.target_location = (np.random.randint(self.width), np.random.randint(self.height))
            if not self.is_collision(*self.target_location):  # Ensure no collision
                break
                    
        self.robot_positions = {
        agent: (np.random.randint(self.width), np.random.randint(self.height))
        for agent in self.possible_agents
        }

        info = {agent: {} for agent in self.agents}
        return self.get_observations(), info

    def step(self, actions):
        # executes action and updates environment
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for robot, action in actions.items():

            action = np.array(action).flatten()

            x, y = self.robot_positions[robot]

            dx, dy = action

            # move in bounds
            new_x = np.clip(x + dx, 0, self.width - 1)
            new_y = np.clip(y + dy, 0, self.height - 1)

            # check collisions
            if not self.is_collision(new_x, new_y):
                self.robot_positions[robot] = (new_x, new_y)
                exploration_reward = self.multiagent_env.calculate_reward(new_x, new_y)
                self.multiagent_env.mark_visited(new_x, new_y)

            # check for target
            if self.within_radius((new_x, new_y), self.target_location):
                rewards[robot] = 1.0
                terminations = {a: True for a in self.agents}
                break
            else:
                rewards[robot] = -0.001

        # check if all robots are done

        info = {agent: {} for agent in self.agents}

        self.agents = [agent for agent in self.agents if not terminations[agent]]  
        done = any(terminations.values())

        return self.get_observations(), rewards, terminations, truncations, info

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
            if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
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
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(self.width, self.height, 1), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return  # Do nothing if rendering is disabled

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Robot Search Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # White background

        # Draw grid
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray grid

        # Draw obstacles (black)
        for obs_x, obs_y, obs_w, obs_h in self.obstacle_coords:
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),  # Black color for obstacles
                pygame.Rect(obs_x * self.cell_size, obs_y * self.cell_size, obs_w * self.cell_size, obs_h * self.cell_size)
            )

        # Draw target (red)
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),  # Red for the target
            (self.target_location[0] * self.cell_size + self.cell_size // 2, 
             self.target_location[1] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )

        # Draw robots (blue)
        for robot, (x, y) in self.robot_positions.items():
            pygame.draw.circle(
                self.screen,
                (0, 0, 255),  # Blue for robots
                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                self.cell_size // 4
            )

        pygame.display.flip()  # Update the screen
        self.clock.tick(5)  # Limit to 10 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    # # initialize environment
    # env = RobotSearchEnv()
    # observations = env.reset()

    # # run a quick random simulation
    # num_steps = 10
    # for step in range(num_steps):
    #     actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.robot_positions.keys()}
    #     observations, rewards, done, x, info = env.step(actions)

    #     print(f"Step {step + 1}:")
    #     print(f"Actions: {actions}")
    #     print(f"Rewards: {rewards}")
    #     print(f"Done: {done}")

    #     if done["robot_0"]:
    #         print("Simulation finished. Target found!")
    #         break

    env = RobotSearchEnv(render_mode="human")
    obs, _ = env.reset()

    for _ in range(50):  # Run test for 50 steps
        actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.agents}
        obs, rewards, done, trunc, _ = env.step(actions)
        env.render()  #

        if done["robot_0"] or done["robot_1"] or done["robot_2"]:
            break

    env.close()