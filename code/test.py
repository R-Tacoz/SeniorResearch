import time
import numpy as np
import torch
from scipy.spatial import KDTree
device = torch.device('cpu')

n_nodes = 5
com_range = 2
positions = torch.ones((5,2))

n = 10_000
# print(np.cross(a,b))
start = time.perf_counter()
for _ in range(n):
    
    edge_index = [
        [i, j]
        for i in range(n_nodes)
        for j in range(i,n_nodes)
        if (
            i != j and 
            torch.norm(positions[i] - positions[j]) < com_range
        )
    ]
            
print(edge_index)
print(time.perf_counter() - start)

start = time.perf_counter()
for _ in range(n):
    
    # Create all possible combinations of node indices
    node_indices = torch.arange(n_nodes, device=device)
    source = node_indices.repeat_interleave(n_nodes)
    target = node_indices.repeat(n_nodes)
    
    # Calculate distances between all pairs of nodes
    source_positions = positions[source]
    target_positions = positions[target]
    distances = torch.norm(source_positions - target_positions, dim=-1)
    
    # Filter out self-loops and pairs beyond communication range
    mask = (source != target) & (distances < com_range)
    
    # Get valid edges
    valid_edges = torch.stack([source[mask], target[mask]], dim=0)
    
    # Handle the case when no edges are found
    if valid_edges.size(1) == 0:
        edge_index = torch.zeros((2, 1), device=device)
    else:
        edge_index = valid_edges
            
print(edge_index)
print(time.perf_counter() - start)


# c = c.to(device)
# d = d.to(device)
# torch.cuda.synchronize
# start = time.perf_counter()
# for _ in range(n):
    
#     c.cross(d, dim=1)
            
# torch.cuda.synchronize()
# print(time.perf_counter() - start)

        
