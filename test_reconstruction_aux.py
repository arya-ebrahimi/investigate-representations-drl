import torch
import gymnasium as gym
from PIL import Image
from core.agent import *
import hydra
from core.utils import *
import numpy as np

# TEST RECONSTRUCTED OBSERVATION OF RECONSTRUCT AUXILIARY TASK


@hydra.main(config_path="config", config_name="transfer_config.yaml", version_base=None)
def main(args):
    model_path = args.model_path
    if args.use_aux == 'virtual-reward-1':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=1)
    elif args.use_aux == 'virtual-reward-5':
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode, virtual_goal=2)
    else:
        env = gym.make('core:MazEnv-v0', goal_mode=args.goal_mode)
    
    model = Agent(env=env, args=args)
    
    model.target_net.load_state_dict(torch.load(model_path))
    model.policy_net.load_state_dict(torch.load(model_path))
            
    state, _ = env.reset()
    images = []
    
    state = state.transpose((2, 0, 1))
    action = model.target_net(torch.tensor(state, device=model.device))[0].argmax().item()

    numpy_image = np.array(model.target_net(torch.tensor(state, device=model.device))[1].squeeze().cpu().detach().numpy())
    numpy_image = (numpy_image/255.0).round()*255.0
    numpy_image = np.array(numpy_image, dtype=np.uint8)
    
    image = Image.fromarray(numpy_image.transpose((1, 2, 0)), "RGB")
    image.show()

    

if __name__ == "__main__":
    main()