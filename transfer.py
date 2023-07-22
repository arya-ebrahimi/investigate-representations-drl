import torch
import gymnasium as gym
from PIL import Image
from core.agent import *
import hydra
from core.utils import *
import numpy as np




@hydra.main(config_path="config", config_name="transfer_config.yaml", version_base=None)
def main(args):
    model_path = '.models/fixed_reward_aux_model.pt'
    env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode)
    model = Agent(env=env, args=args)
    

    model.target_net.load_state_dict(torch.load(model_path))
    model.policy_net.load_state_dict(torch.load(model_path))
    
    d_s = np.load('d_s.npy') + 0.001
    d_v = np.load('d_v.npy')
    
    d = d_v / d_s
    
    # print(d_v.shape)
    # print(d_s+0.001)
    L_max = d.max()
    sum = 0
    for i in range(1000):
        for j in range(i):
            sum+= d[i, j]
    L_rep = sum * 2 /(1000 * 999)    
    
    print(1-(L_rep/L_max))
    
    # The first 8 params are weights and biases of representation network
    # for i, param in enumerate(model.policy_net.parameters()):
    #     if i < 8:
    #         param.requires_grad = False
    
    # model.train()

if __name__ == "__main__":
    main()