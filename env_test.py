import numpy as np
from utils.agents import RandomAgent, MLPAgent, ConvAgent
from utils.envs import MainEnv, PathEnv

def main():

    # # initialize environment
    # env = RobotSearchEnv()
    # observations = env.reset()

    # # run a quick random simulation
    # num_steps = 10
    # for step in range(num_steps):
    #     actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.robot_positions.keys()}
    #     observations, rewards, done, x, info = env.step(actions)

    #     print(f"Step {step + 1}:")
    #     print(f"Actions: {actions}")
    #     print(f"Rewards: {rewards}")
    #     print(f"Done: {done}")

    #     if done["robot_0"]:
    #         print("Simulation finished. Target found!")
    #         break

    # env = MainEnv(
    #         num_robots=3, 
    #         width=20, 
    #         height=20, 
    #         target_location=(8, 8), 
    #         lidar_range = 2,
    #         camera_range = 0, 
    #         render_mode = "human", 
    #         seed = None, 
    #         num_obstacles = 20,
            
    #         options = None
    #     )
    
    env = PathEnv(framerate=10)
    #obs, _ = env.reset()

    for _ in range(5000):
        actions = {robot: np.random.uniform(-1, 1, size=(2,)) for robot in env.agents}
        #actions = {robot: np.array([0.5, 0]) for robot in env.agents}
        obs, rewards, done, trunc, _ = env.step(actions)
        
        #print(f"{env.robot_positions['robot_0'][0]:.2f}, {env.robot_positions['robot_0'][1]:.2f}")
        env.render()

        if any(done.values()):
        #if done["robot_0"] or done["robot_1"] or done["robot_2"]:
            break

    env.close()
    

if __name__ == "__main__":
    main()