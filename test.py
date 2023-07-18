import torch
import gymnasium as gym
from PIL import Image
from core.agents.agent2 import *

model_path = '/home/arya/Desktop/Investigate-Representations-DeepRL/.models/pytorch_2023-04-26_00:48.pt'
env = gym.make('core:MazEnv-v0')
model = Agent(env=env)
model.target_net.load_state_dict(torch.load(model_path))

state, _ = env.reset()
images = []
for i in range(100):
    state = state.transpose((2, 0, 1))
    action = model.target_net(torch.tensor(state, device=model.device)).argmax().item()

    image = Image.fromarray(state.transpose((1, 2, 0)), "RGB")
    images.append(image)
    print(action)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    done = terminated or truncated
    if done:
        break
    state = next_state

images[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    