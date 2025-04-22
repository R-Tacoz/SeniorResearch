import numpy as np
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
import shapely
from shapely.geometry import LineString, Point, box
from scipy.spatial import distance
from scipy.spatial import KDTree
from utils.agents import RandomAgent, MLPAgent, ConvAgent

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

class MainEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "robot_search_v0"} 
    

    def __init__(
        self, 
        num_robots=3, 
        width=20, 
        height=20, 
        target_location=(8, 8), 
        lidar_range = 2,
        camera_range=2, 
        render_mode = None, 
        seed = None, 
        num_obstacles = 6,
        framerate = 10,
        options = None,
    ):
        super().__init__()
        self.multiagent_env = MultiAgentEnv()
        #parameters
        self.render_mode = render_mode
        self.num_robots = num_robots
        self.env_width = width
        self.env_height = height
        
        self.num_obstacles = num_obstacles
        self.obstacle_width = 3 # units are cells for now
        self.obstacle_height = 3
        
        self.target_location = target_location
        
        self.robot_width = 0.5
        self.robot_height = 0.5
        self.lidar_range = lidar_range
        self.camera_range = camera_range
        
        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.framerate = framerate

        # generating obstacles
        self.obstacle_coords = []
        self.generate_obstacles(self.num_obstacles, 2, 2)

        # action and observation spaces
        # self.action_spaces = {agent: Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent in self.possible_agents}
        # self.observation_spaces = {agent: Box(low=0, high=1, shape=(width, height, 1), dtype=np.float32) for agent in self.possible_agents}
        
        # randomize robot positions
        self.robot_positions = {f"robot_{i}": (np.random.randint(width), np.random.randint(height)) for i in range(num_robots)}

        #pygame render initialization
        self.cell_size = 50 
        self.window_size = (self.env_width * self.cell_size, self.env_height * self.cell_size)
        self.screen = None  
        self.clock = None  

#    @override
    def reset(self, seed=None, options = None):
        # resets the environment to its initial state.
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        self.multiagent_env = MultiAgentEnv()

        # self.obstacle_coords = []
        # self.generate_obstacles(self.num_obstacles, self.obstacle_width, self.obstacle_height)

        # while True:
        #     self.target_location = (np.random.randint(self.width), np.random.randint(self.height))
        #     if not self.is_collision(*self.target_location):  # Ensure no collision
        #         break
                    
        self.robot_positions = {
            agent: (np.random.randint(self.env_width), np.random.randint(self.env_height))
            for agent in self.possible_agents
        }

        info = {agent: {} for agent in self.agents}
        return self.get_observations(), info

#    @override
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
            new_x = np.clip(x + dx, 0, self.env_width - 1)
            new_y = np.clip(y + dy, 0, self.env_height - 1)

            # check collisions
            if not self.is_collision(new_x, new_y):
                self.robot_positions[robot] = (new_x, new_y)
                
                exploration_reward = self.multiagent_env.calculate_reward(new_x, new_y)
                self.multiagent_env.mark_visited(new_x, new_y)

            # check for target
            if self.within_radius((new_x, new_y), self.target_location):
                rewards[robot] = 100.0
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

    @staticmethod
    def generate_rays(position, heading, n_rays=180, max_range=10.0):
        """Generate LineString rays for creating the LiDAR scans

        Args:
            position (_type_): _description_
            heading (_type_): _description_
            n_rays (int, optional): _description_. Defaults to 180.
            max_range (float, optional): _description_. Defaults to 10.0.

        Returns:
            _type_: _description_
        """
        angles = np.linspace(-np.pi, np.pi, n_rays) + heading
        dx = np.cos(angles) * max_range
        dy = np.sin(angles) * max_range
        ray_starts = np.repeat([position], n_rays, axis=0)
        ray_ends = ray_starts + np.stack((dx, dy), axis=-1)
        
        return [LineString([start, end]) for start, end in zip(ray_starts, ray_ends)]

    def make_observation(self, position):
        heading = 0
        n_rays = 4
        
        rays: list[LineString] = MainEnv.generate_rays(
            np.array(position), heading, n_rays, self.lidar_range)
        
        scan = np.full(n_rays, self.lidar_range, dtype=np.float64)
        
        for i, ray in enumerate(rays):
            for obstacle in self.obstacle_coords:
                x,y,w,h = obstacle
                obstacle = box(x, y, x + w, y + h)
                intersection = ray.intersection(obstacle)
                if not intersection.is_empty:
                    dist = Point(position).distance(intersection)
                    if dist < scan[i]:
                        scan[i] = dist
                        
        # # observation around a robot's position
        # x, y = position
        # obs = np.zeros((self.width, self.height, 1), dtype=np.float32)
        # for i in range(self.width):
        #     for j in range(self.height):
        #         if self.within_radius((x, y), (i, j)) and not self.is_collision(i, j):
        #             obs[i, j, 0] = 1.0
        
        observations = scan
        return observations

    def within_radius(self, pos1, pos2):
        #for checking if target is within the radius of the robot
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) <= self.camera_range

    def is_collision(self, x, y):
        # check if robot collides with obstacle
        for obs_x, obs_y, obs_w, obs_h in self.obstacle_coords:
            # print(obs_x, obs_y, obs_w, obs_h, end=' | ')
            # print(obs_x <= x <= obs_x + obs_w, end='/')
            # print(obs_y <= y <= obs_y + obs_h)
            if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
                # print("FOUND COLLISION------------------------------------")
                return True
        # print("NO COLLISION")
        return False

    def generate_obstacles(self, num_obstacles, obs_width, obs_height):
        # generates the obstacles
        obstacles = []
        for i in range(num_obstacles):
            while True:
                obs_x = np.random.randint(0, self.env_width - obs_width + 1)
                obs_y = np.random.randint(0, self.env_height - obs_height + 1)
                new_obstacle = (obs_x, obs_y, obs_width, obs_height)
                if not any(self.is_collision(x, y) for x in range(obs_x, obs_x + obs_width) for y in range(obs_y, obs_y + obs_height)):
                    self.obstacle_coords.append(new_obstacle)
                    break
    
    # @override
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(self.env_width, self.env_height, 1), dtype=np.float32)

    # @override
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # @override
    def render(self):
        if self.render_mode != "human":
            return  # Do nothing if rendering is disabled

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Robot Search Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # White background

        fps = self.clock.get_fps()
        pygame.display.set_caption(f"FPS: {fps:.2f}")
        
        # Draw grid
        for x in range(self.env_width):
            for y in range(self.env_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray grid

        # Draw LiDAR Range
        for robot, (x, y) in self.robot_positions.items():
            pygame.draw.circle(
                self.screen,
                (120, 120, 180),
                (x * self.cell_size, y * self.cell_size),
                self.cell_size * self.lidar_range
            )

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
            x_screen, y_screen = x*self.cell_size, y*self.cell_size
            bot_width, bot_height = self.robot_width*self.cell_size, self.robot_height*self.cell_size
            pygame.draw.rect(
                self.screen,
                (0, 0, 255), # Blue for robots
                pygame.Rect(
                    x_screen - bot_width // 2,
                    y_screen - bot_height // 2,
                    bot_width,
                    bot_height,
                ),
            )
            
            # pygame.draw.circle(
            #     self.screen,
            #     (0, 0, 255),  # Blue for robots
            #     (x * self.cell_size, y * self.cell_size),
            #     self.cell_size // 4
            # )
            

        pygame.display.flip()  # Update the screen
        self.clock.tick(self.framerate)  # Limit framerate

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
class PathEnv(MainEnv):
    def __init__(self, framerate=5):
        super().__init__(
            num_robots=1, 
            width=20, 
            height=20, 
            target_location=(19, 19), 
            lidar_range=2,
            camera_range=0, 
            render_mode = "human", 
            seed = None, 
            num_obstacles=0,
            framerate=framerate,
            options = None
            )
        
        self.obstacle_coords = [
            (0, 0, 20, 3),
            (0, 5, 14, 15),
            (17, 2, 3, 15),
        ]
        
        self.robot_positions = {f"robot_{i}": (0, 3.5) for i in range(self.num_robots)}
        
class OpenEnv(MainEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.obstacle_coords = []