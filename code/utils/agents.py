import time
import math
import numpy as np
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
from scipy.spatial import distance
from scipy.spatial import KDTree
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

class RandomAgent():
    def __init__(self):
        pass
    
class MLPAgent(ActorCriticPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class AgentCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, lidar_rays_count: int = 90):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.lidar_rays_count = lidar_rays_count
        self.others_size = observation_space.shape[0] - lidar_rays_count - 2
        
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 4, kernel_size=5, stride=2, padding=0),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_input_channels, lidar_rays_count)
                # torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + self.others_size, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        lidar_scan = observations[:,:self.lidar_rays_count]
        scan_features = self.cnn(lidar_scan.unsqueeze(dim=1))
        
        non_scan = observations[:,self.lidar_rays_count:-2]
        
        new_obs = torch.cat((scan_features, non_scan), dim=-1)
        
        out = self.linear(new_obs)
        
        return out
      
class ConvAgent(ActorCriticCnnPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class AgentGNN(BaseFeaturesExtractor):
    agent_pos_dims = 2

    def __init__(
        self, 
        observation_space: spaces.Box, 
        features_dim: int = 128,
        n_conv_features: int = 16,
        message_size: int = 64, 
        lidar_rays_count: int = 90,
        communication_range: float = 5,
    ):
        super().__init__(observation_space, features_dim)
        
        
        
        self.n_conv_features = n_conv_features
        self.lidar_rays_count = lidar_rays_count
        self.others_size = observation_space.shape[0] - lidar_rays_count - self.agent_pos_dims
        self.communication_range = communication_range
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, n_conv_features, kernel_size=5, stride=2, padding=0),
            # nn.MaxPool1d(kernel_size=3, stride=3),
            nn.AdaptiveAvgPool1d(output_size=1), # bc gpt said so
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     n_flatten = self.cnn(torch.zeros(1, 1, lidar_rays_count)
        #         # torch.as_tensor(observation_space.sample()[None]).float()
        #     ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_conv_features + self.others_size, message_size), 
            nn.ReLU(),
        )
        
        # self.gnn = GCNConv(message_size, features_dim)
        self.gnn = GATConv(message_size, features_dim, heads=2, concat=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:   
        device = obs.device
        
        lidar_scan = obs[:,:self.lidar_rays_count]
        non_scan = obs[:,self.lidar_rays_count:-self.agent_pos_dims]
        positions = obs[:,-self.agent_pos_dims:]
        
        graph_edge_index = self.construct_edge_index(positions, device)
        
        conv_features = self.cnn(lidar_scan.unsqueeze(dim=1))
        
        x = torch.cat((conv_features, non_scan), dim=-1)  
        x = self.linear(x)
        x = self.gnn(x, graph_edge_index)
        
        return x
    
    def construct_edge_index(self, positions, device):
        n_nodes = positions.size(0)
        
        # Create all possible combinations of node indices
        node_indices = torch.arange(n_nodes, device=device)
        source = node_indices.repeat_interleave(n_nodes)
        target = node_indices.repeat(n_nodes)
        
        # Calculate distances between all pairs of nodes
        source_positions = positions[source]
        target_positions = positions[target]
        distances = torch.norm(source_positions - target_positions, dim=-1)
        
        # Filter out self-loops and pairs beyond communication range
        mask = (source != target) & (distances < self.communication_range)
        
        # Get valid edges
        valid_edges = torch.stack([source[mask], target[mask]], dim=0)
        
        # Handle the case when no edges are found
        if valid_edges.size(1) == 0:
            edge_index = torch.zeros((2, 1), device=device)
        else:
            edge_index = valid_edges
            
        return edge_index
    
class GNNPolicy(ActorCriticCnnPolicy):
    
    def __init__(self, *args, **kwargs):
        self.gcn1 = GCNConv()