import numpy as np
import gymnasium as gym
from PIL import Image
from core.agents.agent2 import *
import torch

# DOWN = 0
# RIGHT = 1
# UP = 2
# LEFT = 3

env = gym.make('core:MazEnv-v0')
env.reset()
# state, reward, _, _, _ = env.step(0)
# state = env.get_state_image(state)

# device = 'cuda'

# tensor = torch.tensor(state, device='cuda')
# net = Network().to(device=device)

# a = net(tensor)
# print(a.max().item())
agent = Agent(env=env)
agent.train()


# state = env.get_state_image()
# print(state.shape)

