import time
import numpy as np
import torch
device = torch.device('cuda')

a = np.random.randn(300_000,3)
b = np.random.randn(300_000,3)
c = torch.Tensor(a)
d = torch.Tensor(b)


n = 1_000_000
# print(np.cross(a,b))
start = time.perf_counter()
for _ in range(n):
    
    a = 0
    # np.cross(a,b)
            
print(time.perf_counter() - start)

start = time.perf_counter()
for _ in range(n):
    
    a = time.perf_counter()
    # c.cross(d, dim=1)
            
print(time.perf_counter() - start)


# c = c.to(device)
# d = d.to(device)
# torch.cuda.synchronize
# start = time.perf_counter()
# for _ in range(n):
    
#     c.cross(d, dim=1)
            
# torch.cuda.synchronize()
# print(time.perf_counter() - start)