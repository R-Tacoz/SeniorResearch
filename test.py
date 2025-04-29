import time
import numpy as np
from scipy.spatial import distance


def distance_to_nearest_visited(coords, set):
    dist = min([
        distance.euclidean(coords, visited_coords) 
        for visited_coords in set
    ])
        
    return dist

def np_dist(coords, set):
    dist = np.min(np.linalg.norm(set - coords, axis=-1))
    return dist


arr = np.arange(1000)
val = 500

n = 10

start = time.perf_counter()
for _ in range(n):
    
    visiteds = list()
    for i in range(300):
        coords = (i, i+1)
        visiteds.append(coords)
        
        np_dist(np.array([coords]), np.array(visiteds))
        
        # distance_to_nearest_visited(coords, visiteds)
            
print(time.perf_counter() - start)

start = time.perf_counter()
for _ in range(n):
    
    visiteds = np.empty((0,2))
    for i in range(300):
        coords = np.array([[i,i+1]])
        visiteds = np.append(visiteds, coords, axis=0)
        
        np_dist(coords, visiteds)
            
print(time.perf_counter() - start)