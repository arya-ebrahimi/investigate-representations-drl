import numpy as np
import gymnasium as gym
from PIL import Image


DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3


env = gym.make('core:MazEnv-v0')
env.reset()


image = Image.fromarray(env.get_state_image(), 'RGB')
image.save('img.png')

