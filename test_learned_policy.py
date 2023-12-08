import torch
import gymnasium as gym
from PIL import Image
from core.agent import *
import hydra

from core.utils import pixel_replication

# TEST LEARNED POLICIES AND CREATE GIF OF ONE EPISODE


@hydra.main(config_path="config", config_name="transfer_config.yaml", version_base=None)
def main(args):
    model_path = args.model_path
    env = gym.make('core:MazEnv-v0', goal_mode=0)
    model = Agent(env=env, args=args)
    

    model.target_net.load_state_dict(torch.load(model_path))
    model.policy_net.load_state_dict(torch.load(model_path))
    
    state, _ = env.reset()
    images = []
    for i in range(100):
        state = state.transpose((2, 0, 1))
        action = model.target_net(torch.tensor(state, device=model.device))[0].argmax().item()

        print(model.target_net(torch.tensor(state, device=model.device))[0])
        
        image = Image.fromarray(pixel_replication(state.transpose((1, 2, 0)), 24), "RGB")
        images.append(image)
        print(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        if done:
            break
        state = next_state

    images[0].save('out1.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    



if __name__ == "__main__":
    main()