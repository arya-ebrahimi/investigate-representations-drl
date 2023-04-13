import torch
import gymnasium as gym
from PIL import Image


env = gym.make('core:MazEnv-v0')
model = torch.load('.models/pytorch_2023-04-13_22:37.pt')

state, _ = env.reset()
images = []
for i in range(10):
    action = model.policy_net(torch.tensor([state.transpose(2, 0, 1)], device=model.device)).argmax().item()
    image = Image.fromarray(state, "RGB")
    images.append(image)
    print(action)
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    done = terminated or truncated
    state = next_state

images[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    