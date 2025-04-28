import time
import math
import multiprocessing as mp # horrible idea
import numba 
from numba import jit
import numpy as np
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
from scipy.spatial import distance, KDTree
from utils.agents import RandomAgent, MLPAgent, ConvAgent

coords_t = tuple[float, float]
id_t = str
EPSILON = 1e-2 # for division by distance in reward
LIDAR_RAY_COUNT = 180

class MainEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "robot_search_v0"}
    colors = {
        'bg': (255,255,255),
        'grid': (200,200,200),
        'lidar_range': (120,120,180),
        'camera_range': (100,100,180),
        'success_range': (80, 80, 180),
        'obstacle': (0,0,0),
        'target': (255,0,0),
        'robot': (0,0,255),
        'visiteds': (0,80,0),
    }
    resets_i = 0
    obs_r = None

    def __init__(
        self, 
        num_robots: int = 3, 
        width: int = 20, 
        height: int = 20, 
        target_location: tuple | None = (8, 8), 
        lidar_range: float = 5,
        camera_range: float = 8, 
        success_range: float = 1,
        render_mode: str | None = None, 
        seed: object = None, 
        num_obstacles: int = 6,
        framerate: int = 10,
        options: object = None,
    ):
        """init

        Args:
            num_robots (int, optional): _description_. Defaults to 3.
            width (int, optional): _description_. Defaults to 20.
            height (int, optional): _description_. Defaults to 20.
            target_location (tuple | None, optional): _description_. Defaults to (8, 8).
            lidar_range (float, optional): _description_. Defaults to 2.
            camera_range (float, optional): _description_. Defaults to 2.
            success_range (float, optional): _description_. Defaults to 1.
            render_mode (str | None, optional): _description_. Defaults to None.
            seed (object, optional): _description_. Defaults to None.
            num_obstacles (int, optional): _description_. Defaults to 6.
            framerate (int, optional): _description_. Defaults to 10.
            options (object, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.multiagent_env = None
        
        # init params
        self.render_mode = render_mode
        self.num_robots: int = num_robots
        self.env_width: float = width
        self.env_height: float = height
        
        self.num_obstacles: int = num_obstacles
        self.obstacle_width: float = 3 # units are cells for now
        self.obstacle_height: float = 3
        
        self.target_location: coords_t = target_location
        self.randomize_target: bool = True if self.target_location is None else False
        
        self.robot_width: float = 0.5
        self.robot_height: float = 0.5
        self.lidar_range: float = lidar_range
        self.lidar_ray_count: int = LIDAR_RAY_COUNT
        self.camera_range: float = camera_range
        self.success_range: float = success_range
        
        # simulation environment data
        self.env_box = box(0,0,self.env_width, self.env_height)
        self.env_boundary_vecs = []
        
        # TODO: when the dust settles, we only need one obstacle data variable
        self.obstacle_coords: list[coords_t] = [] # basic calculation
        self.obstacle_coords_points: list[tuple[float]] = [] # wtf
        self.obstacle_edge_vectors: list[np.ndarray[float]] = [] # idek
        self.obstacles: list[Polygon] = [] # ray intersection
        
        self.obstacle_centers: np.ndarray = [] # faster ray intersection
        self.obstacle_tree: KDTree = None
        self.approx_obs_radius: float = (self.obstacle_height + self.obstacle_width) * math.sqrt(2)
        # TODO: look into quadtree and K-D-trees for obstacle querying
        # ^^^ :) maybe not. lets recheck if we increase env size past 20x20 and 8 obstacles 
        
        self.possible_agents: list[id_t] = [f"robot_{i}" for i in range(num_robots)]
        self.robot_positions: dict[id_t, coords_t] = {}
        self.robot_boxes: dict[id_t, Polygon] = {}
        self.robot_box_centers: dict[id_t, np.ndarray] = {}
        
        self.robot_observation_space = Box(
            low=0., high=self.lidar_range, 
            shape=(self.lidar_ray_count + 2 + 2,), # + camera detection + last velocity
            dtype=np.float32
            )
        self.robot_action_space = Box(
            low=-1.0, high=1.0, 
            shape=(2,), dtype=np.float32
            )
        
        angles = np.linspace(-np.pi, np.pi, self.lidar_ray_count)
        dx = np.cos(angles) * self.lidar_range
        dy = np.sin(angles) * self.lidar_range
        self.lidar_ray_displacements = np.stack((dx, dy), axis=-1)
        self.lidar_scan_buffer: np.ndarray = np.zeros(self.lidar_ray_count, dtype=np.float64)
        
        self.ticks_elapsed = 0
        self.framerate = framerate
    
        # agent data
        # TODO: in actual implementation, these are agent data that are stored in each agent
        self.robot_last_velocities: dict[id_t, tuple[float,float]] = {}
        
        self.sparse_visited_coords: list[coords_t] = []
        self.visiteds_min_dist: float = 2 * self.robot_width
             
        #pygame render initialization
        self.active = False
        self.sorted_range_color_pairs = sorted(
            [(self.lidar_range, self.colors['lidar_range']), 
             (self.camera_range, self.colors['camera_range']),
             (self.success_range, self.colors['success_range'])], 
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
        #print("resets", self.resets_i, end='\n')
        # resets the environment to its initial state.
        self.active = True
        self.ticks_elapsed = 0
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        self.sparse_visited_coords = []

        self.regenerate_obstacles(self.num_obstacles, self.obstacle_width, 
                                  self.obstacle_height)
        
        self.obstacle_centers = np.array([
            obstacle.centroid.coords[0] for obstacle in self.obstacles
        ])  # shape (num_obstacles, 2)
        
        # self.obstacle_tree = KDTree(self.obstacle_centers)

        if self.randomize_target:
            while True:
                self.target_location = self.get_random_coord()
                if not self.is_collision(self.target_location):  # Ensure no collision
                    break
                  
        self.robot_positions = {}
        for agent_id in self.possible_agents:
            while True:
                coords = self.get_random_coord(in_grid=False)
                if not self.is_collision(coords):
                    self.robot_positions[agent_id] = coords
                    self.robot_box_centers[agent_id] = np.array(coords)
                    self.sparse_visited_coords.append(coords)
                    break
                
            self.robot_last_velocities[agent_id] = (0,0)

        observations = {id: self.get_observations(id)[0] for id in self.agents}
        self.obs_r = observations
        info = {id: {} for id in self.agents}
        
        return observations, info

#    @override
    def step(self, actions):
        # executes action and updates environment
        self.ticks_elapsed += 1
        
        observations = {a: None for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        info = {a: {} for a in self.agents}

        move_time = 0
        collision_time = 0
        visiteds_time = 0
        reward_time = 0
        obs_time = 0

        for agent_id, action in actions.items():
            
            t0 = time.perf_counter()
            # read action and current state
            action = np.array(action).flatten()
            dx, dy = action

            x, y = self.robot_positions[agent_id]
            
            # move in bounds
            new_x = np.clip(x + dx, 0, self.env_width)
            new_y = np.clip(y + dy, 0, self.env_height)
            new_coords: coords_t = (new_x, new_y)
            
            move_time += (t1:=time.perf_counter()) - t0

            # update position if no collision
            attempted_collision = self.is_collision(agent_id=agent_id, shape=Point(new_coords))
            acceleration = 0
            
            collision_time += (t0:=time.perf_counter()) - t1
            
            if not attempted_collision:
                self.robot_positions[agent_id] = new_coords
                
                self.robot_boxes[agent_id] = box(
                    x - self.robot_width/2, y - self.robot_height/2, 
                    x + self.robot_width/2, y + self.robot_height/2)
                
                for id, robot_box in self.robot_boxes.items():
                    self.robot_box_centers[id] = np.array(robot_box.centroid.coords[0])
                
                # calculate acceleration as ||dv||
                acceleration = distance.euclidean(self.robot_last_velocities[agent_id], (dx,dy))
                
                move_time += (t1:=time.perf_counter()) - t0

                # update visited points
                dist_to_closest_visited = self.distance_to_nearest_visited(new_coords)
                if dist_to_closest_visited > self.visiteds_min_dist:
                    self.sparse_visited_coords.append(new_coords)
                    
            else:
                dist_to_closest_visited = self.distance_to_nearest_visited((x, y))
            visiteds_time += (t0:=time.perf_counter()) - t1
                     
            # get observations    
            # print('ha')   
            observations[agent_id], obs_data = self.get_observations(agent_id)
            
            target_dist, target_in_sight = obs_data
            
            if not attempted_collision: # bc observations include last velocity
                self.robot_last_velocities[agent_id] = (dx, dy)
            
            obs_time += (t1:=time.perf_counter()) - t0
            
            # calculate reward
            rewards[agent_id] = self.calc_reward(agent_id, new_coords, attempted_collision, target_dist, target_in_sight, dist_to_closest_visited, acceleration)
            
            reward_time += (t0:=time.perf_counter()) - t1
      
            # check if target is found (the robot has to drive to it)
            if target_dist < self.success_range:
                # terminate all agents upon task completion
                terminations = {a: True for a in self.agents}
                self.agents = []
                break
            
        move_time *= 1E3
        collision_time *= 1E3
        visiteds_time *= 1E3
        reward_time *= 1E3
        obs_time *= 1E3
        
        # print('step')
        # print(f"m: {move_time:.4f} c: {collision_time:.4f} v: {visiteds_time:.4f} r: {reward_time:.4f} o: {obs_time:.4f}", end='\r')  

        return observations, rewards, terminations, truncations, info

    def get_random_coord(self, in_grid=True) -> tuple:
        if in_grid:
            return (np.random.randint(self.env_width), 
                    np.random.randint(self.env_height))
        else: 
            return (np.random.random() * self.env_width,
                    np.random.random() * self.env_height)

    def generate_rays(self, position, heading, n_rays=180, max_range=10.0) -> list[LineString]:
        """Generate LineString rays for creating the LiDAR scans

        Args:
            position (_type_): _description_
            heading (_type_): _description_
            n_rays (int, optional): _description_. Defaults to 180.
            max_range (float, optional): _description_. Defaults to 10.0.

        Returns:
            _type_: _description_
        """
        
        ray_starts = np.repeat([position], n_rays, axis=0)
        
        # this will be needed if we allow the robots to rotate
        # angles = np.linspace(-np.pi, np.pi, n_rays) + heading
        # dx = np.cos(angles) * max_range
        # dy = np.sin(angles) * max_range
        # ray_ends = ray_starts + np.stack((dx, dy), axis=-1)
        
        ray_ends = ray_starts + self.lidar_ray_displacements
        
        rays = [LineString([start, end]) for start, end in zip(ray_starts, ray_ends)]
        
        return rays

    def fast_ray_cast(self, origin):
        origin = np.array(origin)
        scan = self.lidar_scan_buffer
        scan.fill(self.lidar_range)
        
        for i, center in enumerate(self.obstacle_centers):
            if np.linalg.norm(center - origin) + self.approx_obs_radius > self.lidar_range:
                continue
            
            for p1, p2, edge_vec in self.obstacle_edge_vectors[i]:
                for j, ray_dir in enumerate(self.lidar_ray_displacements/self.lidar_range):
                    rxs = np.cross(ray_dir, edge_vec)
                    
                    if abs(rxs) < 1e-10:
                        continue
                    
                    t = np.cross(p1 - origin, edge_vec) / rxs
                    s = np.cross(p1 - origin, ray_dir) / rxs
                    
                    if 0 <= t <= self.lidar_range and 0 <= s <= 1:
                        scan[j] = min(scan[j], t)
                        
        # TODO: include env walls and other agents as vectorized intersection checks
                        
        return scan          
    

    @staticmethod
    def lidar_worker(args):
        print('work')
        rays_chunk, agent_position, agent_id, obstacle_centers, obstacles, robot_boxes, robot_box_centers, env_box, lidar_range = args
        lidar_scan_chunk = np.full((len(rays_chunk),), lidar_range, dtype=np.float64)
        agent_xy = np.array([agent_position.x, agent_position.y])

        for i, ray in enumerate(rays_chunk):
            # Check environment walls first
            intersection = ray.intersection(env_box.boundary)
            if not intersection.is_empty:
                dist = agent_position.distance(intersection)
                if dist < lidar_scan_chunk[i]:
                    lidar_scan_chunk[i] = dist

            # Vectorized obstacle filtering
            delta_obs = obstacle_centers - agent_xy
            dists_obs = np.linalg.norm(delta_obs, axis=1)
            close_obs_idx = np.where(dists_obs < lidar_range)[0]

            for idx in close_obs_idx:
                obstacle = obstacles[idx]
                intersection = ray.intersection(obstacle)
                if not intersection.is_empty:
                    dist = agent_position.distance(intersection)
                    if dist < lidar_scan_chunk[i]:
                        lidar_scan_chunk[i] = dist

            # Vectorized robot filtering
            for robot_id, robot_box in robot_boxes.items():
                if robot_id == agent_id:
                    continue
                delta_robot = robot_box_centers[robot_id] - agent_xy
                dist_robot = np.linalg.norm(delta_robot)
                if dist_robot > lidar_range:
                    continue

                intersection = ray.intersection(robot_box)
                if not intersection.is_empty:
                    dist = agent_position.distance(intersection)
                    if dist < lidar_scan_chunk[i]:
                        lidar_scan_chunk[i] = dist
                        
        print('done')

        return lidar_scan_chunk

    def get_observations(self, agent_id: id_t) -> tuple[np.ndarray, list]:
        heading = 0
        coords = self.robot_positions[agent_id]
        position = Point(coords)
        
        # LiDAR scan
        # TODO: test scanning every few ticks
        # nearby_obstacles_indices = self.obstacle_tree.query_ball_point(
        #     coords, self.lidar_range + self.approx_obs_radius)
        
        rays = self.generate_rays(
            coords, heading, self.lidar_ray_count, self.lidar_range)
        
        # lidar_scan = self.lidar_scan_buffer
        # lidar_scan.fill(self.lidar_range)
        lidar_scan = self.fast_ray_cast(coords)
        
        # TODO: invert loop order
        
        for i, ray in enumerate(rays):
            # walls
            intersection = ray.intersection(self.env_box.boundary)
            if not intersection.is_empty:
                dist = position.distance(intersection)
                if dist < lidar_scan[i]:
                    lidar_scan[i] = dist
            
        # # obstacles
        # for obstacle in self.obstacles:
        # # for idx in nearby_obstacles_indices:
        #     # obstacle = self.obstacles[idx]
        #     if position.distance(obstacle) > self.lidar_range:
        #         continue
            
        #     for i, ray in enumerate(rays):
        #         intersection = ray.intersection(obstacle)
        #         if not intersection.is_empty:
        #             dist = position.distance(intersection)
        #             if dist < lidar_scan[i]:
        #                 lidar_scan[i] = dist
            
        # other agents
        for id in self.robot_boxes:
            if id == agent_id or position.distance(self.robot_boxes[id]) > self.lidar_range:
                continue
            
            for i, ray in enumerate(rays):
                intersection = ray.intersection(self.robot_boxes[id])
                if not intersection.is_empty:
                    dist = position.distance(intersection)
                    if dist < lidar_scan[i]:
                        lidar_scan[i] = dist
                        
        lidar_scan /= self.lidar_range # normalize
        
        # Camera detection
        camera_detection = np.array([1, 0]) # default if no detection is camera range (normalized), zero heading
        target_dist = distance.euclidean(coords, self.target_location)
        in_sight = False
        if target_dist < self.camera_range:
            # check no obstacles block view
            in_sight = True
            sightline = LineString([coords, self.target_location])
            for obstacle in self.obstacles:
                if sightline.intersects(obstacle):
                    in_sight = False
                    break
                
            if in_sight:
                x = self.target_location[0] - coords[0]
                y = self.target_location[1] - coords[1]
                target_heading = math.atan2(y,x) # assume robot is facing right where heading=0
                
                camera_detection[0] = target_dist / self.camera_range
                camera_detection[1] = target_heading / math.pi
                
        # Kinematic information
        last_velocity = np.array(self.robot_last_velocities[agent_id])
        
        # Displacement history
        # TODO: maybe displacement vector to average of the visiteds
                 
        data = [target_dist, in_sight] # extra data for reward calculation that robots can't observe
        observations = np.concatenate([lidar_scan, camera_detection, last_velocity], axis=0)
        return observations, data

    def is_collision(
        self, 
        coords: coords_t=None,
        agent_id: str=None,
        shape: Geometry=None, # lets you pass in a Polygon
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
            shape = Point(*coords)
            
        # if isinstance(shape, Point):
        #     if 0 >= shape.x or shape.x >= self.env_width or 0 >= shape.y or shape.y >= self.env_height:
        #         return True
        
        # TODO: check if this matters (it's a slowdown and also affects reward calculation)
        # if self.env_box.boundary.intersects(shape):
        #     return True
            
        for obstacle in self.obstacles:
            if obstacle.intersects(shape):
                return True
        
        for id in self.robot_boxes:
            if id != agent_id and self.robot_boxes[id].intersects(shape):
                return True
            
        return False
    
    # @staticmethod
    # @jit(nopython=True)
    def distance_to_nearest_visited(self, coords):
        dist = min([
            distance.euclidean(coords, visited_coords) 
            for visited_coords in self.sparse_visited_coords
        ])
        
        
        
        # min_dist = float('inf')
        # for visited in sparse_visited_coords:
        #     dx = coords[0] - visited[0]
        #     dy = coords[1] - visited[1]
        #     dist = (dx*dx + dy*dy)**0.5
        #     if dist < min_dist:
        #         min_dist = dist
        # return min_dist
        
        return dist
    
    def calc_reward(
        self, 
        agent_id: id_t, 
        coords: coords_t, 
        attempted_collision: bool, 
        target_dist: float, 
        target_in_sight: bool,
        nearest_visited_dist: float,
        acceleration: float,
    ) -> float:
        
        reward = 0.0
        
        # time penalty
        reward += -0.01 * self.ticks_elapsed
        
        # acceleration penalty
        reward += -0.5 * acceleration
        
        # collision penalty
        if attempted_collision:
            reward += -10
        
        # exploration reward
        if nearest_visited_dist > self.visiteds_min_dist:
            reward += 5
            
        # TODO: maybe reward distance to average of visited points? rn only looks at nearest
        
        # re-exploration penalty
        reward += -1/(nearest_visited_dist + EPSILON) # don't want penalty to exceed success reward
        
        # target sight reward
        if target_in_sight:
            reward += 50
            reward += 1/(target_dist + EPSILON)
            
        # success reward
        if target_dist < self.success_range:
            reward += 5000
            
        return reward
  
    def regenerate_obstacles(self, num_obstacles, obs_width, obs_height):
        # regenerates the obstacles
        self.obstacle_coords = []
        for i in range(num_obstacles):
            while True:
                coord = self.get_random_coord()
                new_obstacle_coords = coord + (obs_width, obs_height)
                
                # test overlapping obstacles
                self.obstacle_coords.append(new_obstacle_coords)
                
                minx, miny = coord
                maxx, maxy = minx + obs_width, miny + obs_height
                self.obstacle_coords_points.append((minx, miny, maxx, maxy))
                self.obstacles.append(box(minx, miny, maxx, maxy))
                break
            
        for minx, miny, maxx, maxy in self.obstacle_coords_points: 
            edges = [
                [(minx, miny), (maxx, miny)],
                [(maxx, miny), (maxx, maxy)],
                [(maxx, maxy), (minx, maxy)],
                [(minx, maxy), (minx, miny)],
            ]
            edge_lord_gooner = []
            for p1, p2 in edges:
                p1, p2 = np.array(p1), np.array(p2)
                edge_vector = p2 - p1
                edge_lord_gooner.append((p1, p2, edge_vector))
                
            self.obstacle_edge_vectors.append(edge_lord_gooner)
 
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

        # Draw Ranges
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
        
        # Draw visited points
        for (x, y) in self.sparse_visited_coords:
            pygame.draw.circle(
                self.screen,
                self.colors['visiteds'],
                (x * self.cell_size, y * self.cell_size),
                self.cell_size // 12,
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
        self.obstacles = []
        for coords in self.obstacle_coords:
            minx, miny = coords
            maxx, maxy = minx + self.obstacle_width, miny + self.obstacle_height
            self.obstacles.append(box(minx, miny, maxx, maxy))
        
        self.robot_positions = {f"robot_{i}": (0, 3.5) for i in range(self.num_robots)}
        
class OpenEnv(MainEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.obstacle_coords = []
        self.obstacles = []