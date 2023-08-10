import numpy as np
import gymnasium as gym
from PIL import Image
from core.agent import *
import torch
import hydra

# DOWN = 0
# RIGHT = 1
# UP = 2
# LEFT = 3

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
    if args.use_aux == 'virtual-reward-1':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=1)
    elif args.use_aux == 'virtual-reward-5':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=2)
    else:
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode)
    
    for i in range(args.runs):
        env.reset()
        agent = Agent(env=env, args=args)
        agent.train()


if __name__ == "__main__":
    main()
