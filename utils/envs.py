import time
import math
import multiprocessing as mp # horrible idea
from numba import jit
from numba.np.extensions import cross2d
import numpy as np
import torch
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box, Discrete
# TODO: atp idt we need shapely stuff anymore. we can clean it out of code when everything works
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
from scipy.spatial import distance, KDTree
from utils.agents import RandomAgent, MLPAgent, ConvAgent

# TODO: eventually move all coords_t usages to np.ndarray
coords_t = tuple[float, float]
id_t = str
EPSILON = 1e-2 # for division by distance in reward
LIDAR_RAY_COUNT = 90

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
        self.env_dims = (self.env_width, self.env_height)
        self.env_boundary_vecs = [ # (p1, p2, edge_vec)
            (np.array([0,0]), np.array([self.env_width, 0]), np.array([self.env_width, 0])),
            (np.array([self.env_width,0]), np.array([self.env_width, self.env_height]), np.array([0, self.env_height])),
            (np.array([self.env_width,self.env_height]), np.array([0, self.env_height]), np.array([-self.env_width, 0])),
            (np.array([0, self.env_height]), np.array([0, 0]), np.array([0, -self.env_height])),
        ]
        
        # TODO: when the dust settles, we only need one obstacle data variable
        self.obstacle_coords: list[coords_t] = [] # basic calculation
        self.obstacle_coords_points: list[tuple[float]] = [] # wtf
        self.obstacle_edge_vectors: list[np.ndarray[float]] = [] # idek
        self.obstacles: list[Polygon] = [] # ray intersection
        
        self.obstacle_centers: np.ndarray = [] # faster ray intersection
        self.obstacle_tree: KDTree = None
        # TODO: approximate obstacle radius for each obstacle as a ndarray
        self.approx_obs_radius: float = np.linalg.norm((self.obstacle_width/2, self.obstacle_height/2))
        # TODO: look into quadtree and K-D-trees for obstacle querying
        # ^^^ :) maybe not. lets recheck if we increase env size past 20x20 and 8 obstacles 
        
        self.possible_agents: list[id_t] = [f"robot_{i}" for i in range(num_robots)]
        self.agents = self.possible_agents[:]
        
        self.robot_positions: dict[id_t, coords_t] = {}
        self.robot_boxes: dict[id_t, Polygon] = {}
        self.robot_box_centers: dict[id_t, np.ndarray] = {}
        self.robot_box_corners: dict[id_t, np.ndarray] = {}
        self.robot_box_edge_vectors: np.ndarray = {} # all agents have the same edge lengths
        self.approx_robot_box_radius: float = np.linalg.norm((self.robot_width/2, self.robot_height/2))
        
        self.robot_observation_space = Box(
            low=0., high=self.lidar_range, 
            shape=(self.lidar_ray_count + 2 + 2,), # + camera detection + last velocity
            dtype=np.float32
            )
        self.robot_action_space = Box(
            low=-1.0, high=1.0, 
            shape=(2,), dtype=np.float32
            )
        
        self.lidar_ray_indices = np.arange(self.lidar_ray_count)
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.lidar_ray_count)
        self.lidar_ray_directions = np.stack((np.cos(self.lidar_angles), np.sin(self.lidar_angles)), axis=-1)
        self.lidar_ray_displacements = self.lidar_ray_directions * self.lidar_range
        self.lidar_scan_buffer: np.ndarray = np.zeros(self.lidar_ray_count, dtype=np.float64)
        
        self.ticks_elapsed = 0
        self.framerate = framerate
    
        # agent data
        # TODO: in actual implementation, these are agent data that are stored in each agent
        self.robot_last_velocities: dict[id_t, tuple[float,float]] = {}
        
        self.sparse_visited_coords: np.ndarray = None
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
    def reset(self, seed=None, options = None) -> tuple:
        """Initialize all values and re-randomizes obstacles and positions

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple: obs, info
        """
        self.resets_i += 1
        #print("resets", self.resets_i, end='\n')
        self.active = True
        self.ticks_elapsed = 0
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        # regenerate obstacles
        self.regenerate_obstacles()

        # re-randomize target location
        if self.randomize_target:
            while True:
                self.target_location = self.get_random_coord()
                if not self.is_collision(self.target_location):
                    break
                
        # re-randomize agent positions and initialize the visited set
        corners_template = np.array([
            [-self.robot_width/2, -self.robot_height/2],
            [self.robot_width/2, -self.robot_height/2],
            [self.robot_width/2, self.robot_height/2],
            [-self.robot_width/2, self.robot_height/2],
        ])
                  
        self.robot_positions = {}
        self.sparse_visited_coords = np.empty((0,2))
        for agent_id in self.possible_agents:
            coords = None
            while True:
                coords = self.get_random_coord(in_grid=False)
                if not self.is_collision(coords): 
                    break
                
            point = np.array(coords)
            self.robot_positions[agent_id] = coords
            self.robot_box_centers[agent_id] = point
            self.robot_box_corners[agent_id] = point + corners_template
            
            self.sparse_visited_coords = np.append(self.sparse_visited_coords, [point], axis=0)
            self.robot_last_velocities[agent_id] = (0,0)

        self.robot_box_edge_vectors = np.array([
            [self.robot_width, 0],
            [0, self.robot_height],
            [-self.robot_width, 0],
            [0, -self.robot_height],
        ])

        # create observations and info to return
        observations = {id: self.get_observations(id)[0] for id in self.agents}
        self.obs_r = observations
        info = {id: {} for id in self.agents}
        
        return observations, info

#    @override
    def step(self, actions) -> tuple:
        """Executes all actions and updates the environment

        Args:
            actions (_type_): _description_

        Returns:
            tuple: _description_
        """
        
        # initialization
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
            ds = action

            x, y = self.robot_positions[agent_id]
            
            # move in bounds
            new_x = np.clip(x + dx, 0.01, self.env_width-0.01)
            new_y = np.clip(y + dy, 0.01, self.env_height-0.01)
            new_coords: coords_t = (new_x, new_y)
            new_coords_a = np.array(new_coords)
            
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
                
                self.robot_box_centers[agent_id] += ds
                self.robot_box_corners[agent_id] += ds
                
                # calculate acceleration as ||dv||
                acceleration = distance.euclidean(self.robot_last_velocities[agent_id], (dx,dy))
                
                move_time += (t1:=time.perf_counter()) - t0

                # update visited points
                dist_to_closest_visited = self.distance_to_nearest_visited(new_coords_a)
                if dist_to_closest_visited > self.visiteds_min_dist:
                    self.sparse_visited_coords = np.append(self.sparse_visited_coords, [new_coords_a], axis=0)
                    
            else:
                dist_to_closest_visited = self.distance_to_nearest_visited(np.array([x, y]))
            visiteds_time += (t0:=time.perf_counter()) - t1
                     
            # get observations     
            observations[agent_id], obs_data = self.get_observations(agent_id)
            
            target_dist, target_in_sight = obs_data
            
            if not attempted_collision: # bc observations include last velocity
                self.robot_last_velocities[agent_id] = (dx, dy)
            
            obs_time += (t1:=time.perf_counter()) - t0
            
            # calculate reward
            rewards[agent_id] = self.calc_reward(
                agent_id, new_coords, attempted_collision, target_dist, 
                target_in_sight, dist_to_closest_visited, acceleration)
            
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
        
        # print(f"m: {move_time:.3f} c: {collision_time:.3f} v: {visiteds_time:.3f} r: {reward_time:.4f} o: {obs_time:.4f}", end='\r')  

        return observations, rewards, terminations, truncations, info

    def get_observations(self, agent_id: id_t) -> tuple[np.ndarray, list]:
        heading = 0
        coords = self.robot_positions[agent_id]
        
        # LiDAR scan
        # TODO: test scanning every few ticks
        # nearby_obstacles_indices = self.obstacle_tree.query_ball_point(
        #     coords, self.lidar_range + self.approx_obs_radius)
        
        lidar_scan = self.fast_ray_cast(coords, agent_id)          
        lidar_scan /= self.lidar_range # normalize
        
        t0 = time.perf_counter() * 1000
        
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
        
        t2 = time.perf_counter() * 1000
        
        # print(f"cam: {t2-t0:.3f}", end='\r')
        
        # Displacement history
        # TODO: maybe displacement vector to average of the visiteds
                 
        data = [target_dist, in_sight] # extra data for reward calculation 
        observations = np.concatenate([lidar_scan, camera_detection, last_velocity], axis=0)
        return observations, data

    def get_random_coord(self, in_grid=True) -> tuple:
        if in_grid:
            return (np.random.randint(self.env_width), 
                    np.random.randint(self.env_height))
        else: 
            return (np.random.random() * self.env_width,
                    np.random.random() * self.env_height)

    # @jit(nopython=True)
    def fast_ray_cast(self, origin, agent_id=None) -> np.ndarray:
        origin = np.array(origin)
        scan = self.lidar_scan_buffer
        scan.fill(self.lidar_range)
        
        # obstacles
        obstacle_displacements = self.obstacle_centers - origin
        obstacle_distances = np.linalg.norm(obstacle_displacements, axis=1) # broadcasts
        obstacle_distances -= self.approx_obs_radius # can be converted to an array
        close_obstacle_center_indices = (obstacle_distances <= self.lidar_range).nonzero()[0]
        
        t0 = time.perf_counter() * 1000
        for idx in close_obstacle_center_indices:     
            # select rays that face the obstacle
            # this assumes obstacles are all convex. if they aren't, we can add another data variable to indicate it
            displacement = obstacle_displacements[idx]
            angle_to_obstacle = np.arctan2(displacement[1], displacement[0])
            indices = get_fov_mask_indices(self.lidar_angles, angle_to_obstacle, np.pi)
            # indices = self.lidar_ray_indices
            
            for start_corner, _, edge_vec in self.obstacle_edge_vectors[idx]:
                write_edge_intersections(scan, indices, self.lidar_ray_directions, self.lidar_range, origin, edge_vec, start_corner)
                
        t1 = time.perf_counter() * 1000
        
        # environment boundaries          
        for i, (start_corner, p2, edge_vec) in enumerate(self.env_boundary_vecs):            
            perp_dim = (i+1) % 2 # dimension perpendicular to this edge
            if self.lidar_range <= origin[perp_dim] <= self.env_dims[perp_dim] - self.lidar_range:
                continue
            
            angle_to_boundary = i * np.pi/2 - np.pi/2
            indices = get_fov_mask_indices(self.lidar_angles, angle_to_boundary, np.pi)
            # indices = self.lidar_ray_indices
            write_edge_intersections(scan, indices, self.lidar_ray_directions, self.lidar_range, origin, edge_vec, start_corner)

        t2 = time.perf_counter() * 1000
        
        # other robots
        for id_ in self.agents:
            if id_ == agent_id:
                continue
            
            displacement_to_other = self.robot_box_centers[id_] - origin
            dist_to_other_center = np.linalg.norm(displacement_to_other)
            if dist_to_other_center - self.approx_robot_box_radius > self.lidar_range:
                continue
            
            angle_to_robot = np.arctan2(displacement_to_other[1], displacement_to_other[0])
            indices = get_fov_mask_indices(self.lidar_angles, angle_to_robot, np.pi/2)

            for start_corner, edge_vec in zip(self.robot_box_corners[id_], self.robot_box_edge_vectors):
                write_edge_intersections(scan, indices, self.lidar_ray_directions, self.lidar_range, origin, edge_vec, start_corner)
        
        t3 = time.perf_counter() * 1000
        
        # print(f"obs:{t1-t0:.3f}\tbound:{t2-t1:.3f}\trobo:{t3-t2:.3f}", end=" ")
                             
        return scan       

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
    
    def distance_to_nearest_visited(self, coords) -> float:
        dist = np.min(np.linalg.norm(self.sparse_visited_coords - coords, axis=-1))
        
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
  
    def regenerate_obstacles(self, obstacle_coords=None) -> None:
        """Regenerates obstacles

        Args:
            obstacle_coords (_any_, optional): If used, will init obstacles there. If not, will randomize. Defaults to None.
        """
        
        tl_corners = []
        if obstacle_coords is None:
            self.obstacle_coords = []
            for i in range(self.num_obstacles):
                while True:
                    coord = self.get_random_coord()
                    tl_corners.append(coord)
                    new_obstacle_coords = coord + (self.obstacle_width, self.obstacle_height)
                    
                    # test overlapping obstacles
                    self.obstacle_coords.append(new_obstacle_coords)
                    
                    minx, miny = coord
                    maxx, maxy = minx + self.obstacle_width, miny + self.obstacle_height
                    self.obstacle_coords_points.append((minx, miny, maxx, maxy))
                    self.obstacles.append(box(minx, miny, maxx, maxy))
                    break
        else:
            self.obstacle_coords = obstacle_coords
            for (x,y,w,h) in self.obstacle_coords:
                tl_corners.append((x,y))
                maxx, maxy = x + w, y + h
                self.obstacle_coords_points.append((x,y,maxx,maxy))
                self.obstacles.append(box(x,y,maxx,maxy)) 
                
        self.obstacle_centers = np.array(tl_corners) + np.array([self.obstacle_width/2, self.obstacle_height/2])  # shape (num_obstacles, 2)
            
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
    def render(self) -> None:
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
        # print(self.sparse_visited_coords)
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
            
        # Draw LiDAR points
        if self.num_agents <= 2:
            origin = self.robot_positions['robot_0']
            scan = np.repeat(self.lidar_scan_buffer.reshape(-1,1), 2, axis=1)
            for coord in np.multiply(self.lidar_ray_displacements, scan):
                coord[0] += origin[0]
                coord[1] += origin[1]
                pygame.draw.circle(
                    self.screen,
                    (0,255,0),
                    (coord[0] * self.cell_size, coord[1] * self.cell_size),
                    3,
                )
            
        pygame.display.flip()  # Update the screen
        self.clock.tick(self.framerate)  # Limit framerate

        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Quitting")
                self.close()

    def close(self) -> None:
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
        self.reset()
        
    def reset(self):
        self.resets_i += 1
        self.ticks_elapsed = 0
        self.active = True

        self.agents = self.possible_agents[:]

        self.robot_positions = {f"robot_{i}": tuple([0, 3.5]) for i in range(self.num_robots)}
        self.sparse_visited_coords = np.empty((0,2))
        
        for agent_id in self.robot_positions:
            point = self.robot_positions[agent_id]
            self.robot_box_centers[agent_id] = point
            self.sparse_visited_coords = np.append(self.sparse_visited_coords, [point], axis=0)             
            self.robot_last_velocities[agent_id] = (0,0)
        
        obstacle_coords = [
            (0, 0, 20, 3),
            (0, 6, 14, 15),
            (17, 2, 3, 15),
        ]
        self.regenerate_obstacles(obstacle_coords)
        
        observations = {id: self.get_observations(id)[0] for id in self.agents}
        self.obs_r = observations
        info = {id: {} for id in self.agents}
        
        return observations, info
        
class OpenEnv(MainEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.obstacle_coords = []
        self.obstacles = []   


def write_edge_intersections(
    scan: np.ndarray, 
    indices_array: np.ndarray, 
    lidar_ray_directions: np.ndarray, 
    lidar_range: float, 
    origin: np.ndarray, 
    edge_vec: np.ndarray, 
    edge_start: np.ndarray
) -> None:
    """Updates a distance scan using intersection distances to a single other vector

    Args:
        scan (np.ndarray): _description_
        indices_array (np.ndarray): _description_
        lidar_ray_directions (np.ndarray): _description_
        lidar_range (float): _description_
        origin (np.ndarray): _description_
        edge_vec (np.ndarray): _description_
        edge_start (np.ndarray): _description_
    """
    
    ray_directions = lidar_ray_directions[indices_array]
    dists = vector_intersection_distances(origin, ray_directions, edge_vec, edge_start)
    scan[indices_array] = np.min(np.stack((scan[indices_array],dists)), axis=0)

@jit(nopython=True)
def get_fov_mask_indices(
    full_fov_array: np.ndarray, 
    heading: float, 
    fov: float
) -> np.ndarray:
    """Returns the indices of full_fov_array that would be in the FOV in the direction of heading.

    Args:
        full_fov_array (np.ndarray): _description_
        heading (float): _description_
        fov (float): _description_

    Returns:
        np.ndarray: _description_
    """
    delta_angles = np.mod(full_fov_array - heading + np.pi, 2*np.pi) - np.pi
    mask = (delta_angles >= -fov/2) & (delta_angles <= fov/2)
    indices = np.where(mask)[0]
    return indices

def vector_intersection_distances(
    origin, direction_vecs, other_vec, other_vec_start
) -> np.ndarray:
    """Calculates the intersection distances of all vectors in direction_vecs to the other_vec

    Args:
        origin (_type_): Center of the direction vectors.
        direction_vecs (_type_): Unit vectors in directions.
        other_vec (_type_): A singular vector to check
        other_vec_start (_type_): A (vector) that points to the start of the other vector in the same coordinates as origin.

    Returns:
        np.ndarray: Distances in the same length as direction_vecs. Those that don't intersect are assigned infinity.
    """
    vec_start_disp = other_vec_start - origin
    rxs = np.cross(direction_vecs, other_vec) + 1e-5 # get magnitudes only
    
    # t*direction_vec reaches intersection
    t = np.cross(vec_start_disp, other_vec) / rxs
    
    # vec_start_disp + s*other_vec reaches intersection
    s = np.cross(vec_start_disp, direction_vecs) / rxs
    
    # only count intersection if it's in the edge
    mask = (s < 0) | (s > 1)
    
    t[mask] = np.Infinity
    
    return t

@jit(nopython=True, parallel=True)
def vector_intersection_distance_jit(
    origin: np.ndarray, 
    direction_vec: np.ndarray, 
    other_vec: np.ndarray, 
    other_vec_start: np.ndarray
) -> float:
    vec_start_disp = other_vec_start - origin
    # rxs = np.cross(direction_vec, other_vec)
    rxs = cross2d(direction_vec, other_vec)
    
    if abs(rxs.item()) < 1e-10:
        return -1.0
    
    t = cross2d(vec_start_disp, other_vec) / rxs
    s = cross2d(vec_start_disp, direction_vec) / rxs
    
    if 0 <= s <= 1:
        return t.item()
    else:
        return -1.0