import time
import numpy as np
# from utils.envs_torch import MainEnv
from utils.envs import MainEnv, PathEnv
from pynput import keyboard

FRAMERATE = 320 # also equals tickrate
SIM_LENGTH = 100000 # frames/ticks
MAX_VELO = 4 # cells / s
MAX_DISP = 2
x_input, y_input = 0,0
reset_sim = False

def on_press(key):
    global x_input, y_input, reset_sim
    if key == keyboard.Key.up:      y_input = -1
    elif key == keyboard.Key.down:  y_input = 1
    elif key == keyboard.Key.right: x_input = 1
    elif key == keyboard.Key.left:  x_input = -1
    elif isinstance(key, keyboard.KeyCode):
            if key.char == 'r': reset_sim = True

def on_release(key):
    global x_input, y_input
    if key == keyboard.Key.up:      y_input = 0
    elif key == keyboard.Key.down:  y_input = 0
    elif key == keyboard.Key.right: x_input = 0
    elif key == keyboard.Key.left:  x_input = 0

def main():
    global reset_sim
    listener = keyboard.Listener(on_press, on_release)
    listener.start()
    
    np.set_printoptions(linewidth=200)
    
    # env = PathEnv(framerate=FRAMERATE)
    n_robots = 1
    env = MainEnv(
        num_robots = n_robots, 
        width = 18, 
        height = 18, 
        target_location = None, 
        # lidar_range = 5,
        camera_range = 8, 
        success_range = 0,
        num_obstacles=8,        
        framerate=FRAMERATE, 
        render_mode='human'
    )
    
    t0 = time.perf_counter()
    for _ in range(SIM_LENGTH):
        dt = time.perf_counter() - t0
        t0 += dt
        
        velo_command = np.array([x_input, y_input], dtype=np.float64)
        if (x_input | y_input) != 0:
            velo_command /= np.linalg.norm(velo_command) / (MAX_VELO*dt) # normalize magnitude
        
        np.clip(velo_command, -MAX_DISP, MAX_DISP, out=velo_command)
        
        actions = {
            robot: velo_command 
            if i == n_robots-1 else np.zeros((2,), dtype=np.float64) 
            for i,robot in enumerate(env.agents)
        }
        obs, rewards, terms, truncs, info = env.step(actions)
        
        # print(f"{obs['robot_0']} | {rewards['robot_0']}", end='\r')
        #print(f"{env.robot_positions['robot_0'][0]:.2f}, {env.robot_positions['robot_0'][1]:.2f}", end='\r')
        env.render()
        
        if reset_sim:
            obs = env.reset()
            reset_sim = False

        if not env.active:
            break

    env.close()
    

if __name__ == "__main__":
    main()