import time

start = time.perf_counter()

n = 100_000
for _ in range(n):
    
    a = [i*10 for i in range(100)]
    b = min(a)
    
print(time.perf_counter() - start)

start = time.perf_counter()
for _ in range(n):
    
    b = min([i*10 for i in range(100)])
            
print(time.perf_counter() - start)