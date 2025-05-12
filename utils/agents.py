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
from torch_geometric.nn import GCNConv

class RandomAgent():
    def __init__(self):
        pass
    
class MLPAgent(ActorCriticPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class AgentCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, lidar_rays_count: int = 90):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        self.lidar_rays_count = lidar_rays_count
        self.others_size = observation_space.shape[0] - lidar_rays_count
        
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
        
        non_scan = observations[:,self.lidar_rays_count:]
        
        new_obs = torch.cat((scan_features, non_scan), dim=-1)
        
        out = self.linear(new_obs)
        
        return out
      
class ConvAgent(ActorCriticCnnPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class GNNPolicy(ActorCriticCnnPolicy):
    
    def __init__(self, *args, **kwargs):
        self.gcn1 = GCNConv()
    
class GNNFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, *args, **kwargs):
        pass