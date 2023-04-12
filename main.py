import numpy as np
import gymnasium as gym
from PIL import Image
from core.agents.network import Network
import torch

net = Network()
print(net)


input = torch.randn(3, 15, 15)
out = net(input)
print(out)

# DOWN = 0
# RIGHT = 1
# UP = 2
# LEFT = 3


# env = gym.make('core:MazEnv-v0')
# env.reset()


# image = Image.fromarray(env.get_state_image(), 'RGB')
# image.save('img.png')

