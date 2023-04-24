import torch
import gymnasium as gym
from PIL import Image
from core.agents.agent import *

env = gym.make('core:MazEnv-v0')
model = Agent(env=env)
model.target_net.load_state_dict(torch.load('/home/arya/Desktop/Investigate-Representations-DeepRL/.models/pytorch_2023-04-24_22:41.pt'))

state, _ = env.reset()
images = []
for i in range(100):
    action = model.policy_net(torch.tensor([state.transpose(2, 0, 1)], device=model.device)).max(1)[1].item()
    # print(action)
    # action = env.action_space.sample()
    image = Image.fromarray(state, "RGB")
    images.append(image)
    print(action)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    done = terminated or truncated
    state = next_state

images[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    