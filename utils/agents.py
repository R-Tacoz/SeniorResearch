import time
import math
import numpy as np
import shapely
from shapely.geometry import LineString, Point, box, Polygon
from shapely import Geometry
from scipy.spatial import distance
from scipy.spatial import KDTree
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

class RandomAgent():
    def __init__(self):
        pass
    
class MLPAgent(ActorCriticPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class ConvAgent(ActorCriticCnnPolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    