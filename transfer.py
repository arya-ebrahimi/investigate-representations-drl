import torch
import gymnasium as gym
from PIL import Image
from core.agent import *
import hydra
from core.utils import *
import numpy as np


@hydra.main(config_path="config", config_name="transfer_config.yaml", version_base=None)
def main(args):
    model_path = args.model_path
    if args.use_aux == 'virtual-reward-1':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=1)
    elif args.use_aux == 'virtual-reward-5':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=2)
    else:
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode)
    
    for i in range(args.runs):
        model = Agent(env=env, args=args)
        

        model.target_net.load_state_dict(torch.load(model_path))
        model.policy_net.load_state_dict(torch.load(model_path))
        
        # The first 8 params are weights and biases of representation network
        for i, param in enumerate(model.policy_net.parameters()):
            if i < 8:
                param.requires_grad = False
        
        model.train()

if __name__ == "__main__":
    main()