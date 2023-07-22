import torch
import gymnasium as gym
from PIL import Image
from core.agent import *
import hydra

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
    args.use_aux = 'reward'
    model_path = '.models/fixed_reward_aux_model.pt'
    env = gym.make('core:MazEnv-v0', goal_mode=1)
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