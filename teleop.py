import numpy as np
from utils.envs import MainEnv, PathEnv
from pynput import keyboard

FRAMERATE = 1500 # also equals tickrate
SIM_LENGTH = 10000 # frames/ticks
MAX_VELO = 0.2
x_input, y_input = 0,0

def on_press(key):
    global x_input, y_input
    if key == keyboard.Key.up:      y_input = -1
    elif key == keyboard.Key.down:  y_input = 1
    elif key == keyboard.Key.right: x_input = 1
    elif key == keyboard.Key.left:  x_input = -1

def on_release(key):
    global x_input, y_input
    if key == keyboard.Key.up:      y_input = 0
    elif key == keyboard.Key.down:  y_input = 0
    elif key == keyboard.Key.right: x_input = 0
    elif key == keyboard.Key.left:  x_input = 0

def main():
    listener = keyboard.Listener(on_press, on_release)
    listener.start()

    #env = PathEnv(framerate=FRAMERATE)
    env = MainEnv(framerate=FRAMERATE, render_mode='human')

    for _ in range(SIM_LENGTH):
        velo_command = np.array([x_input, y_input], dtype=np.float64)
        if (x_input | y_input) != 0:
            velo_command /= np.linalg.norm(velo_command) / MAX_VELO # normalize magnitude
        
        actions = {robot: velo_command if i == 0 else np.zeros((2,)) for i,robot in enumerate(env.agents)}
        obs, rewards, terms, truncs, info = env.step(actions)
        
        print(obs['robot_0'], end='\r')
        #print(f"{env.robot_positions['robot_0'][0]:.2f}, {env.robot_positions['robot_0'][1]:.2f}", end='\r')
        env.render()

        if any(terms.values()) or not env.active:
            break

    env.close()
    

if __name__ == "__main__":
    main()