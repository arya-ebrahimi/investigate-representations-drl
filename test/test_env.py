import gymnasium as gym
import numpy as np
from PIL import Image


def pixel_replication(img, k):
    res = np.repeat(img, k, axis=0)
    res = np.repeat(res, k, axis=1)
    return res

def plot_virtual_tasks(virtual_task=1):
    env = gym.make('core:MazEnv-v0', virtual_goal=virtual_task)
    image = Image.fromarray(pixel_replication(env.return_virtual_goals(), 18), "RGB")

    image.save('vf5.png')

def plot_transfer_tasks():
    env = gym.make('core:MazEnv-v0')

    state, _ = env.reset()
    
    state[9, 11] = np.array([0, 200, 0])
    state[1, 4] = np.array([0, 200, 0])
    image = Image.fromarray(pixel_replication(state, 18), "RGB")

    image.save('transfer_tasks.png')
    
    
def main():
   plot_transfer_tasks() 
    
if __name__ == '__main__':
    main()