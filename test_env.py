import gymnasium as gym
import numpy as np
from PIL import Image
from core.utils import pixel_replication

# TEST ENVIRONMENTS IMAGES


NAME_TRANSFER = 'transfer_tasks.png'  # name of transfer task image to be saved
NAME_VIRTUAL = 'vf5.png'  # name of virtual task image to be saved  



def plot_virtual_tasks(virtual_task=1):
    
    '''
    creates an image of environment with virtual tasks defined in MazEnv
    
    inputs:
        -virtual_task: which virtual task to create (1 or 2)
        
    '''
    
    env = gym.make('core:MazEnv-v0', virtual_goal=virtual_task)
    image = Image.fromarray(pixel_replication(env.return_virtual_goals(), 18), "RGB")

    image.save(NAME_VIRTUAL)

def plot_transfer_tasks():
    
    '''
    creates an image of environment with transfer tasks defined in MazEnv    
    '''
    
    env = gym.make('core:MazEnv-v0')

    state, _ = env.reset()
    
    state[9, 11] = np.array([0, 200, 0])
    state[1, 4] = np.array([0, 200, 0])
    image = Image.fromarray(pixel_replication(state, 18), "RGB")

    image.save(NAME_TRANSFER)
    
    
def main():
   plot_transfer_tasks() 
    
if __name__ == '__main__':
    main()