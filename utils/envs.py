import numpy as np
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
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
        # self.visited_points.append((agent_x, agent_y))
        # hella memory leak
        return

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
    colors = {
        'bg': (255,255,255),
        'grid': (200,200,200),
        'lidar_range': (120,120,180),
        'camera_range': (100,100,180),
        'obstacle': (0,0,0),
        'target': (255,0,0),
        'robot': (0,0,255),
    }
    resets_i = 0

    def __init__(
        self, 
        num_robots: int = 3, 
        width: int = 20, 
        height: int = 20, 
        target_location: tuple | None = (8, 8), 
        lidar_range: float = 2,
        camera_range: float = 2, 
        render_mode: str | None = None, 
        seed: object = None, 
        num_obstacles: int = 6,
        framerate: int = 10,
        options: object = None,
    ):
        super().__init__()
        self.multiagent_env = None
        
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
        self.lidar_ray_count = 8
        self.camera_range = camera_range
        
        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.robot_positions: dict[str,tuple] = {}
        self.robot_boxes: dict[str, Polygon] = {}
        
        self.robot_observation_space = Box(
            low=0., high=self.lidar_range, 
            shape=(self.lidar_ray_count,), 
            dtype=np.float32
            )
        self.robot_action_space = Box(
            low=-1.0, high=1.0, 
            shape=(2,), dtype=np.float32
            )
        
        self.timestep = 0
        self.framerate = framerate

        # generating obstacles
        self.env_box = box(0,0,self.env_width, self.env_height)
        self.obstacle_coords = []
        self.obstacles: list[Polygon] = []
             
        #pygame render initialization
        self.active = False
        self.sorted_range_color_pairs = sorted(
            [(self.lidar_range, self.colors['lidar_range']), 
             (self.camera_range, self.colors['camera_range'])], 
            reverse=True
        )
        self.cell_size = 50 
        self.window_size = (self.env_width * self.cell_size, 
                            self.env_height * self.cell_size)
        self.screen = None  
        self.clock = None  
        
        # generate the environment
        self.reset()

#    @override
    def reset(self, seed=None, options = None):
        self.resets_i += 1
        print("resets", self.resets_i, end='\n')
        # resets the environment to its initial state.
        self.active = True
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        self.multiagent_env = MultiAgentEnv()

        self.regenerate_obstacles(self.num_obstacles, self.obstacle_width, 
                                  self.obstacle_height)

        if self.target_location is None:
            while True:
                self.target_location = self.get_random_coord()
                if not self.is_collision(*self.target_location):  # Ensure no collision
                    break
                    
        self.robot_positions = {}
        for agent_id in self.possible_agents:
            while True:
                coords = self.get_random_coord()
                if not self.is_collision(*coords):
                    self.robot_positions[agent_id] = coords
                    break

        observations = self.get_observations()
        info = {agent: {} for agent in self.agents}
        
        return observations, info

#    @override
    def step(self, actions):
        # executes action and updates environment
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for agent_id, action in actions.items():

            action = np.array(action).flatten()

            x, y = self.robot_positions[agent_id]
            self.robot_boxes[agent_id] = box(
                x - self.robot_width/2, y - self.robot_height/2, 
                x + self.robot_width/2, y + self.robot_height/2)

            dx, dy = action

            # move in bounds
            new_x = np.clip(x + dx, 0, self.env_width)
            new_y = np.clip(y + dy, 0, self.env_height)
            
            # TODO: include last robot velocity as part of observation
            # TODO: penalize acceleration (turning)
            

            # check collisions
            if not self.is_collision(agent_id=agent_id, shape=Point(new_x, new_y)):
                self.robot_positions[agent_id] = (new_x, new_y)
                
                exploration_reward = self.multiagent_env.calculate_reward(new_x, new_y)
                self.multiagent_env.mark_visited(new_x, new_y)

            # check for target
            if MainEnv.dist((x, y), self.target_location) < self.camera_range:
                rewards[agent_id] = 100.0
                terminations = {a: True for a in self.agents}
                break
            else:
                rewards[agent_id] = -0.001

        observations = self.get_observations()
        info = {agent: {} for agent in self.agents}
        
        # deactivate terminated robots
        self.agents = [agent for agent in self.agents if not terminations[agent]]  

        return observations, rewards, terminations, truncations, info

    def get_random_coord(self, in_grid=True) -> tuple:
        if in_grid:
            return (np.random.randint(self.env_width), 
                    np.random.randint(self.env_height))
        else: 
            return (np.random.random() * self.env_width,
                    np.random.random() * self.env_height)

    def get_observations(self):
        #all observations for every robot
        observations = {}
        for agent_id in self.robot_positions:
            observations[agent_id] = self.get_robot_observation(agent_id)
        return observations

    def generate_rays(self, position, heading, n_rays=180, max_range=10.0):
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
        #ray_ends = np.clip(ray_ends, 0, self.env_width) # TODO: clip for rectangular env
        
        rays = [LineString([start, end]) for start, end in zip(ray_starts, ray_ends)]
        
        return rays

    def get_robot_observation(self, agent_id: str):
        heading = 0
        coords = self.robot_positions[agent_id]
        position = Point(coords)
        
        rays: list[LineString] = self.generate_rays(
            np.array(coords), heading, self.lidar_ray_count, self.lidar_range)
        
        scan = np.full(self.lidar_ray_count, self.lidar_range, dtype=np.float64)
        
        for i, ray in enumerate(rays):
            intersection = ray.intersection(self.env_box.boundary)
            if not intersection.is_empty:
                dist = position.distance(intersection)
                if dist < scan[i]:
                    scan[i] = dist
            
            for obstacle in self.obstacles:
                #if position.distance(obstacle) > self.lidar_range
                
                intersection = ray.intersection(obstacle)
                if not intersection.is_empty:
                    dist = position.distance(intersection)
                    if dist < scan[i]:
                        scan[i] = dist
            
            for id in self.robot_boxes:
                if id == agent_id:
                    continue
                
                intersection = ray.intersection(self.robot_boxes[id])
                if not intersection.is_empty:
                    dist = position.distance(intersection)
                    if dist < scan[i]:
                        scan[i] = dist
                 
        observations = scan
        return observations

    @staticmethod
    def dist(coord_1, coord_2):
        return np.linalg.norm(np.array(coord_1) - np.array(coord_2))

    def is_collision(
        self, 
        x: float=None, 
        y: float=None, 
        agent_id: str=None,
        shape: Geometry=None,
    ) -> bool:
        """Checks if a point or an agent collides with an obstacle or an agent

        Args:
            x (float, optional): coordinates. Defaults to None.
            y (float, optional): coordinates. Defaults to None.
            agent_id (str, optional): _description_. Defaults to None.

        Returns:
            bool:
        """
        if shape is None:
            shape = Point(x,y)
            
        for obstacle in self.obstacles:
            if obstacle.intersects(shape):
                return True
        
        for id in self.robot_boxes:
            if id != agent_id and self.robot_boxes[id].intersects(shape):
                return True
            
        return False

    def regenerate_obstacles(self, num_obstacles, obs_width, obs_height):
        # regenerates the obstacles
        self.obstacle_coords = []
        for i in range(num_obstacles):
            while True:
                # obs_x = np.random.randint(0, self.env_width - obs_width + 1)
                # obs_y = np.random.randint(0, self.env_height - obs_height + 1)
                # new_obstacle_coords = (obs_x, obs_y, obs_width, obs_height)
                coord = self.get_random_coord()
                new_obstacle_coords = coord + (obs_width, obs_height)
                
                # test overlapping obstacles
                self.obstacle_coords.append(new_obstacle_coords)
                
                minx, miny = coord
                maxx, maxy = minx + obs_width, miny + obs_height
                self.obstacles.append(box(minx, miny, maxx, maxy))
                break
                
                if not any(self.is_collision(x, y) for x in range(obs_x, obs_x + obs_width) for y in range(obs_y, obs_y + obs_height)):
                    self.obstacle_coords.append(new_obstacle_coords)
                    break
    
    # @override
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        agent_type = agent_id[:5]
        if agent_type == "robot":
            return self.robot_observation_space
        elif agent_type == "drone":
            return Box()
        else:
            raise Exception("what kinda agent is this")
        
    # @override
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        agent_type = agent_id[:5]
        if agent_type == "robot":
            return self.robot_action_space
        elif agent_type == "drone":
            return Box()
        else:
            raise Exception("what kinda agent is this")

    # @override
    def render(self):
        if self.render_mode != "human":
            return  # Do nothing if rendering is disabled

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Robot Search Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors['bg'])

        fps = self.clock.get_fps()
        pygame.display.set_caption(f"FPS: {fps:.2f}")
        
        # Draw grid
        for x in range(self.env_width):
            for y in range(self.env_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)  # Light gray grid

        # Draw LiDAR and Camera Range
        for range_, color in self.sorted_range_color_pairs:
            for robot, (x, y) in self.robot_positions.items():
                pygame.draw.circle(
                    self.screen,
                    color,
                    (x * self.cell_size, y * self.cell_size),
                    self.cell_size * range_
                )

        # Draw obstacles (black)
        for obs_x, obs_y, obs_w, obs_h in self.obstacle_coords:
            pygame.draw.rect(
                self.screen,
                self.colors['obstacle'],  # Black color for obstacles
                pygame.Rect(obs_x * self.cell_size, obs_y * self.cell_size, obs_w * self.cell_size, obs_h * self.cell_size)
            )

        # Draw target (red)
        pygame.draw.circle(
            self.screen,
            self.colors['target'],  # Red for the target
            (self.target_location[0] * self.cell_size, 
             self.target_location[1] * self.cell_size),
            self.cell_size // 4
        )
        
        # Draw robots (blue)
        for robot, (x, y) in self.robot_positions.items():
            x_screen, y_screen = x*self.cell_size, y*self.cell_size
            bot_width, bot_height = self.robot_width*self.cell_size, self.robot_height*self.cell_size
            pygame.draw.rect(
                self.screen,
                self.colors['robot'], # Blue for robots
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
            if event.type == pygame.QUIT or \
                event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Quitting")
                self.close()

    def close(self):
        self.active = False
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
            camera_range=0.5, 
            render_mode = "human", 
            seed = None, 
            num_obstacles=0,
            framerate=framerate,
            options = None
            )
        
        self.obstacle_coords = [
            (0, 0, 20, 3),
            (0, 6, 14, 15),
            (17, 2, 3, 15),
        ]
        
        self.robot_positions = {f"robot_{i}": (0, 3.5) for i in range(self.num_robots)}
        
class OpenEnv(MainEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.obstacle_coords = []