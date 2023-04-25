import gymnasium as gym
env = gym.make('core:MazEnv-v0')


state, _ = env.reset()
from PIL import Image
image = Image.fromarray(state, "RGB")
image.show()


