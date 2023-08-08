import gymnasium as gym
env = gym.make('core:MazEnv-v0', virtual_goal=1)


# state = env.return_virtual_goals()
from PIL import Image
# image = Image.fromarray(state, "RGB")
# image.show()

state, _ = env.reset()

print(env.step(1)[4]['virtual-reward'])