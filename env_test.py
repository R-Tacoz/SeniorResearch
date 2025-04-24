import numpy as np
from utils.agents import RandomAgent, MLPAgent, ConvAgent
from utils.envs import MainEnv, PathEnv

def main():
    
    env = MainEnv(
            num_robots=3, 
            width=20, 
            height=20, 
            target_location=None, 
            lidar_range = 2,
            camera_range = 1, 
            render_mode = "human", 
            seed = None, 
            num_obstacles = 20,
            
            options = None
        )
    
    # env = PathEnv(framerate=10)
    num_steps = 500

    for _ in range(num_steps):
        actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.agents}
        observations, rewards, terms, truncs, info = env.step(actions)
        
        #print(f"{env.robot_positions['robot_0'][0]:.2f}, {env.robot_positions['robot_0'][1]:.2f}")
        env.render()

        if any(terms.values()) or not env.active:
            break

    env.close()
    

if __name__ == "__main__":
    main()