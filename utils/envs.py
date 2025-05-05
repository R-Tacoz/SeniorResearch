import sys, time
import math
import multiprocessing as mp # horrible idea
from numba import jit
from numba.np.extensions import cross2d
import numpy as np
import torch
import pygame
import functools
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium.spaces import Box, Discrete
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
# TODO: atp idt we need shapely stuff anymore. we can clean it out of code when everything works
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
from scipy.spatial import distance, KDTree
from utils.agents import RandomAgent, MLPAgent, ConvAgent

# TODO: eventually move all coords_t usages to np.ndarray
coords_t = tuple[float, float]
id_t = str
EPS_REWARD = 1.25 # for division by distance in reward = 0.8 is max weight
EPS_LIDAR = 1e-8
BOUNDS_PAD = 1e-2
LIDAR_RAY_COUNT = 90

class MainEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "robot_search_v0"}
    colors = {
        'bg': (255,255,255),
        'grid': (200,200,200),
        'camera_range': (170, 170, 255),
        'communication_range': (170, 255, 170),
        'lidar_range': (145, 145, 255),
        'success_range': (120, 120, 255),
        'lidar_point': (0,255,0),
        'sightline': (255,150,220),
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
        communication_range: float = 8.2,
        success_range: float = 1,
        render_mode: str | None = None, 
        seed: object = None, 
        num_obstacles: int = 6,
        framerate: int = 10,
        first_init: bool = True,
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
        
        self.seed = seed
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
        self.communication_range: float = communication_range
        self.success_range: float = success_range
        
        self.options = options
        
        # simulation environment data
        # TODO:
        self.env_box = box(0,0,self.env_width, self.env_height)
        
        self.env_dims = np.array([self.env_width, self.env_height])
        self.env_bounds_pad = np.array([BOUNDS_PAD, BOUNDS_PAD])
        self.env_boundary_vecs = [ # (start_point, edge_vec)
            (np.array([0,0]), np.array([self.env_width, 0])),
            (np.array([self.env_width,0]), np.array([0, self.env_height])),
            (np.array([self.env_width,self.env_height]), np.array([-self.env_width, 0])),
            (np.array([0, self.env_height]), np.array([0, -self.env_height])),
        ]
        
        # TODO: when the dust settles, we only need one obstacle data variable
        self.obstacle_coords: list[coords_t] = [] # basic calculation
        self.obstacle_coords_points: list[tuple[float]] = [] # wtf
        self.obstacles: list[Polygon] = [] # ray intersection
        
        self.obstacle_edge_vectors: list[np.ndarray[float]] = [] # idek
        
        self.obstacle_centers: np.ndarray = [] # faster ray intersection
        self.obstacle_tree: KDTree = None
        # TODO: approximate obstacle radius for each obstacle as a ndarray
        self.approx_obs_radius: float = np.linalg.norm((self.obstacle_width/2, self.obstacle_height/2))
        # TODO: look into quadtree and K-D-trees for obstacle querying
        # ^^^ :) maybe not. lets recheck if we increase env size past 20x20 and 8 obstacles 
        
        self.possible_agents: list[id_t] = [f"robot_{i}" for i in range(num_robots)]
        self.agents = self.possible_agents[:]
        
        # TODO:
        self.robot_positions: dict[id_t, np.ndarray] = {}
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
        
        self.robot_max_velocity = 1.0
        self.robot_action_space = Box(
            low=-1.0, high=1.0, 
            shape=(2,), dtype=np.float32
            )
        
        self.lidar_ray_indices = np.arange(self.lidar_ray_count)
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.lidar_ray_count)
        self.lidar_ray_directions = np.stack((np.cos(self.lidar_angles), np.sin(self.lidar_angles)), axis=-1)
        self.lidar_ray_displacements = self.lidar_ray_directions * self.lidar_range
        self.lidar_scan_buffers: dict[id_t, np.ndarray] = {
            id_: np.zeros(self.lidar_ray_count, dtype=np.float64)
            for id_ in self.agents
        }
        
        self.robot_target_sightlines: dict[id_t, np.ndarray | None] = {}
        self.prev_target_dists: dict[id_t, float | None] = {}
        
        self.ticks_elapsed = 0
        self.framerate = framerate
    
        # agent data
        # TODO: in actual implementation, these are agent data that are stored in each agent
        self.robot_last_velocities: dict[id_t, tuple[float,float]] = {}
        
        self.sparse_visited_coords: np.ndarray = None
        self.visiteds_min_dist: float = 0.25 * self.lidar_range
             
        #pygame render initialization
        self.active = False
        self.sorted_range_color_pairs = sorted(
            [(self.lidar_range, self.colors['lidar_range']), 
             (self.camera_range, self.colors['camera_range']),
             (self.success_range, self.colors['success_range']),
             (self.communication_range, self.colors['communication_range'])], 
            reverse=True
        )
        self.cell_size = 50 
        self.window_size = (self.env_width * self.cell_size, 
                            self.env_height * self.cell_size)
        self.screen = None  
        self.clock = None  
        
        # force numba jit-compilation
        dummy = np.zeros(2, dtype=np.float64)
        # get_fov_mask_indices(dummy, 0., 0.)
        vector_intersection_distance_jit(dummy, dummy, dummy, dummy)
        # vector_intersection_distance_jit(dummy, dummy, dummy, dummy)
        
        # generate the environment
        if first_init:
            self.reset(first_reset=True)

#    @override
    def reset(self, seed=None, first_reset=False, options = None) -> tuple:
        """Initialize all values and re-randomizes obstacles and positions

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple: obs, info
        """
        
        # maybe some memory leak causes the fps drops, so just reset everything
        if first_reset:
            self.__init__(
                num_robots = self.num_robots, 
                width = self.env_width, 
                height = self.env_height, 
                target_location = self.target_location, 
                lidar_range = self.lidar_range,
                camera_range = self.camera_range, 
                communication_range = self.communication_range,
                success_range = self.success_range,
                render_mode = self.render_mode, 
                seed = self.seed, 
                num_obstacles = self.num_obstacles,
                framerate = self.framerate,
                first_init = False,
                options = self.options,
            )
            
        self.resets_i += 1
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
        self.robot_box_centers = {}
        self.robot_box_corners = {}
        self.sparse_visited_coords = np.empty((0,2))
        self.robot_last_velocities = {}
        for agent_id in self.possible_agents:
            coords = None
            while True:
                coords = self.get_random_coord(in_grid=False)
                if not self.is_collision(coords): 
                    break
                
            point = np.array(coords)
            self.robot_positions[agent_id] = point
            self.robot_box_centers[agent_id] = point
            self.robot_box_corners[agent_id] = point + corners_template
            
            self.sparse_visited_coords = np.append(self.sparse_visited_coords, [point], axis=0)
            self.robot_last_velocities[agent_id] = (0,0)
            
            self.robot_target_sightlines[agent_id] = None

        self.robot_box_edge_vectors = np.array([
            [self.robot_width, 0],
            [0, self.robot_height],
            [-self.robot_width, 0],
            [0, -self.robot_height],
        ])
        
        
        self.prev_target_dists = {id_:None for id_ in self.agents}

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
        if not self.active:
            raise Exception("Stepping inactive environment")
        
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
            action *= self.robot_max_velocity
            dx, dy = action
            ds = action # = velocity
            dist_moved = np.linalg.norm(ds)

            s0 = self.robot_positions[agent_id]
            x, y = s0
            
            # move
            new_s = s0 + ds
            
            move_time += (t1:=time.perf_counter()) - t0

            # update position if no collision
            # attempted_collision = is_collision_jit(new_s, agent_id, self.env_bounds_pad, self.env_dims, self.obstacle_edge_vectors, self.robot_box_corners)
            attempted_collision = self.is_collision(agent_id=agent_id, shape=Point(new_s))
            acceleration = 0
            
            collision_time += (t0:=time.perf_counter()) - t1
            
            if not attempted_collision:
                # np.clip(
                #     new_s, 
                #     self.env_bounds_pad, 
                #     self.env_dims - self.env_bounds_pad, 
                #     out=new_s
                # )
                # ds = new_s - s0
                self.robot_positions[agent_id] = new_s #- self.robot_positions[agent_id]
                
                self.robot_boxes[agent_id] = box(
                    x - self.robot_width/2, y - self.robot_height/2, 
                    x + self.robot_width/2, y + self.robot_height/2)
                
                self.robot_box_centers[agent_id] += ds
                self.robot_box_corners[agent_id] += ds
                
                # calculate acceleration as ||dv||
                acceleration = distance.euclidean(self.robot_last_velocities[agent_id], (dx,dy))
                
                move_time += (t1:=time.perf_counter()) - t0

                # update visited points
                dist_to_closest_visited = self.distance_to_nearest_visited(new_s)
                if dist_to_closest_visited > self.visiteds_min_dist:
                    self.sparse_visited_coords = np.append(self.sparse_visited_coords, [new_s], axis=0)
                    
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
                agent_id, new_s, attempted_collision, target_dist, 
                target_in_sight, dist_to_closest_visited, acceleration, 
                dist_moved, self.prev_target_dists[agent_id])
            
            self.prev_target_dists[agent_id] = target_dist
            
            reward_time += (t0:=time.perf_counter()) - t1
      
            # check if target is found (the robot has to drive to it)
            if target_dist < self.success_range and not any(terminations.values()):
                # terminate all agents if any one found the target
                # teamwork makes the dream work
                terminations = {a: True for a in self.agents}
                # print("terminating")
                self.agents = []
                self.active=False
                # break
            
        move_time *= 1E3
        collision_time *= 1E3
        visiteds_time *= 1E3
        reward_time *= 1E3
        obs_time *= 1E3
        
        # print("env_step")
        # print(terminations)
        # print(truncations)

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
        disp_to_target = self.target_location - coords
        target_dist = np.linalg.norm(disp_to_target)
        in_sight = False
        
        t1 = time.perf_counter() * 1000
        t2 = t1
        if target_dist < self.camera_range:
            # check no obstacles block view
            in_sight = True
            
            sightline = self.target_location - coords
            for obs_set in self.obstacle_edge_vectors:
                for (start, edge) in obs_set:
                    dist_factor = vector_intersection_distance_jit(coords, sightline, edge, start)
                    if 0 < dist_factor < 1:
                        in_sight = False
                        break
                    
                else:
                    continue
                break
            
            if in_sight:
                self.robot_target_sightlines[agent_id] = sightline
                
                # assume robot is facing right where heading=0
                target_heading = np.arctan2(disp_to_target[1],disp_to_target[0]) 
                
                camera_detection[0] = target_dist / self.camera_range
                camera_detection[1] = target_heading / math.pi
            
            # break sight when an obstacle blocks view
            else:
                self.robot_target_sightlines[agent_id] = None
                
        # break sight when out of range
        else:
            self.robot_target_sightlines[agent_id] = None
                
        # Kinematic information
        last_velocity = np.array(self.robot_last_velocities[agent_id])
        
        t3 = time.perf_counter() * 1000
        
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
        scan = self.lidar_scan_buffers[agent_id]
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
            
            for start_corner, edge_vec in self.obstacle_edge_vectors[idx]:
                write_edge_intersections(scan, indices, self.lidar_ray_directions, self.lidar_range, origin, edge_vec, start_corner)
                
        t1 = time.perf_counter() * 1000
        
        # environment boundaries          
        for i, (start_corner, edge_vec) in enumerate(self.env_boundary_vecs):            
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

    # TODO: numba jit
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

        if coords is None:
            coords = (shape.x, shape.y)
            
            if (
                coords[0] < BOUNDS_PAD or coords[0] > self.env_width - BOUNDS_PAD or 
                coords[1] < BOUNDS_PAD or coords[1] > self.env_height - BOUNDS_PAD
            ):
                return True
            

        # for obstacle in self.obstacles:
        #     if obstacle.intersects(shape):
        #         print('COL')
        #         return True
            
        for obs_ in self.obstacle_edge_vectors:
            tl = obs_[0][0]
            br = obs_[2][0]
            if (
                    tl[0] <= coords[0] <= br[0] and
                    tl[1] <= coords[1] <= br[1]
            ):
                return True
            
        for id_, corners in self.robot_box_corners.items():
            tl = corners[0]
            br = corners[2]
            if id_ != agent_id:
                if (
                    tl[0] <= coords[0] <= br[0] and
                    tl[1] <= coords[1] <= br[1]
                ):
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
        dist_moved: float,
        prev_target_dist: float,
    ) -> float:
        
        reward = 0.0
        
        # time penalty
        reward += -0.01
        
        # velocity reward
        # if dist_moved > 0.1:
        #     reward += 0.5
        
        
        # collision penalty
        if attempted_collision:
            reward += -0.2
        
        # exploration reward
        if nearest_visited_dist > self.visiteds_min_dist:
            reward += 1.5
        else:
            # re-exploration penalty
            reward += -0.5 * math.exp(2 * nearest_visited_dist)
            
        # TODO: maybe reward distance to average of visited points? rn only looks at nearest
        # TODO: informatino gain reward, either number of new cells explored or 
#         visit_counts = np.array([...])  # flat list of visit frequencies to each cell
        # prob = visit_counts / visit_counts.sum()
        # info_gain = entropy(prob)

        # reward += 0.1 * info_gain
        
        # reward += -0.5/(nearest_visited_dist + EPS_REWARD) # don't want penalty to exceed success reward
        
        # target proximity reward
        reward += 0.1/(target_dist + EPS_REWARD)
        
        # target sight reward
        if target_in_sight:
            reward += 2.0
            reward += 1 / (target_dist + EPS_REWARD)
            
            if prev_target_dist is not None:
                delta = prev_target_dist - target_dist
                reward += 0.5 * delta
        else:
            reward += -0.05 * acceleration  # discourage excessive acceleration

        # success reward
        if target_dist < self.success_range:
            reward += 7.0            
            
        return reward
  
    def regenerate_obstacles(self, obstacle_coords=None) -> None:
        """Regenerates obstacles

        Args:
            obstacle_coords (_any_, optional): If used, will init obstacles there. If not, will randomize. Defaults to None.
        """
        
        self.obstacle_coords = [] # basic calculation
        self.obstacle_coords_points = [] # wtf
        self.obstacles = [] # ray intersection
        self.obstacle_edge_vectors = [] # idek
        self.obstaclej_centers = None # faster ray intersection
        
        tl_corners = []
        if obstacle_coords is None:
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
                p1, p2 = np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)
                edge_vector = p2 - p1
                edge_lord_gooner.append((p1, edge_vector))
                
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
        pygame.display.set_caption(f"MAPPO Search Env | FPS: {fps:.2f}")
        
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

        # Draw visited points
        for (x, y) in self.sparse_visited_coords:
            pygame.draw.circle(
                self.screen,
                self.colors['visiteds'],
                (x * self.cell_size, y * self.cell_size),
                self.cell_size // 12,
            )
            
        # Draw sightlines
        for id_, line in self.robot_target_sightlines.items():
            if line is None:
                continue
            
            start = self.robot_positions[id_]
            end = start + line
            pygame.draw.line(
                self.screen,
                self.colors['sightline'],
                start * self.cell_size,
                end * self.cell_size,
                6,
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
            
        # Draw LiDAR points
        for id_, scan in self.lidar_scan_buffers.items():
            origin = self.robot_positions[id_]
            scan = np.repeat(scan.reshape(-1,1), 2, axis=1)
            
            # origin = self.robot_positions[f"robot_{self.num_robots-1}"]
            # scan = np.repeat(self.lidar_scan_buffer.reshape(-1,1), 2, axis=1)
            for coord in np.multiply(self.lidar_ray_displacements, scan):
                coord[0] += origin[0]
                coord[1] += origin[1]
                pygame.draw.circle(
                    self.screen,
                    self.colors['lidar_point'],
                    (coord[0] * self.cell_size, coord[1] * self.cell_size),
                    2,
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

@jit(nopython=True, parallel=True)
def is_collision_jit(
    coords: np.ndarray,
    agent_id: id_t,
    bounds_pad: np.ndarray,
    bounds_dims: np.ndarray,
    obstacles: list,
    all_robot_corners: dict,
) -> bool:
    
    bounds_far_padded = bounds_dims - bounds_pad
    if (
        coords[0] < bounds_pad[0] or coords[0] > bounds_far_padded[0] or 
        coords[1] < bounds_pad[1] or coords[1] > bounds_far_padded[1]
    ):
        return True
        
    for obs_ in obstacles:
        tl = obs_[0][0]
        br = obs_[2][0]
        if (
            tl[0] <= coords[0] <= br[0] and
            tl[1] <= coords[1] <= br[1]
        ):
            return True
        
    for id_, corners in all_robot_corners:
        tl = corners[0]
        br = corners[2]
        if id_ != agent_id and (
            tl[0] <= coords[0] <= br[0] and
            tl[1] <= coords[1] <= br[1]
        ):
            return True
        
    return False

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
    rxs = np.cross(direction_vecs, other_vec) + EPS_LIDAR # get magnitudes only
    
    # t*direction_vec reaches intersection
    t = np.cross(vec_start_disp, other_vec) / rxs
    
    # vec_start_disp + s*other_vec reaches intersection
    s = np.cross(vec_start_disp, direction_vecs) / rxs
    
    # only count intersection if it's in the edge
    mask = (s < 0) | (s > 1) | (t < 0)
    
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
    rxs = cross2d(direction_vec, other_vec) + EPS_LIDAR
    
    # if abs(rxs.item()) < 1e-10:
    #     return np.Inf
    
    t = cross2d(vec_start_disp, other_vec) / rxs
    s = cross2d(vec_start_disp, direction_vec) / rxs
    
    if 0 <= s <= 1:
        return t.item()
    else:
        return np.Inf

class SinglePettingZooVecEnv(VecEnv):
    
    def __init__(self, pettingzoo_env: ParallelEnv):
        """Wraps a single PettingZoo ParallelEnv into a single Stable-Baselines3 VecEnv

        Args:
            pettingzoo_env (_type_): _description_
        """
        self.env: ParallelEnv = pettingzoo_env
        self.agents = self.env.possible_agents

        self.num_agents = len(self.agents)
        self.num_envs = self.num_agents # treat agents as separate envs to allow compatability

        obs_space = self.env.observation_space(self.agents[0])
        act_space = self.env.action_space(self.agents[0])

        self.observation_space = obs_space
        self.action_space = act_space
        
        self.render_mode = None

        super().__init__(num_envs=self.num_envs, observation_space=obs_space, action_space=act_space)

    def reset(self):
        obs, info = self.env.reset()
        obs_array = self._dict_to_array(obs)
        return obs_array

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        action_dict = {agent: self._actions[i] for i, agent in enumerate(self.agents)}
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        
        obs_array = self._dict_to_array(obs)
        reward_array = self._dict_to_array(rewards)
        done_array = self._dict_to_array(terms) | self._dict_to_array(truncs)
        info_array = [infos[agent] for agent in self.agents]
        # reward_array = np.array([[rewards[agent] for agent in self.agents]])
        # done_array = np.array([[terms[agent] or truncs[agent] for agent in self.agents]])
        # info_array = [{} for _ in self.agents]

        return obs_array, reward_array, done_array, info_array

    def close(self):
        self.env.close()

    def _dict_to_array(self, dict_):
        return np.array([dict_[agent] for agent in self.agents])

    def get_attr(self, attr_name: str, indices = None) -> list:
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value, indices = None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) -> list:
        [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices = None) -> list[bool]:
        if wrapper_class is SinglePettingZooVecEnv:
            return [True]
        else:
            return [False]
    