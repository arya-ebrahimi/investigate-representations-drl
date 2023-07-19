import torch
import torch.nn as nn
import os
from pathlib import Path
import random
from tqdm import trange
from itertools import count
import matplotlib.pyplot as plt
import matplotlib
import math
import datetime
from core.utils import *
from core.nn import Network

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class Agent():
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.num_episodes = args.num_episodes
        self.save_ratio=args.save_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.target_update = args.target_update
        self.learning_rate = args.learning_rate
        self.horizon = args.horizon
        self.id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.model_dir = Path('.models')
        self.tau = args.tau
        self.print_ratio = args.print_ratio
        self.use_fta = args.use_fta
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.action_space = env.action_space.n
        
        self.policy_net = Network(self.use_fta).to(self.device)
        self.target_net = Network(self.use_fta).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        
        self.memory = ReplayMemory(1000000)
        
        self.steps_done = 0
        self.reward_in_episode = []
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            
            
    def plot_rewards(self, show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.reward_in_episode, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(rewards_t.numpy())
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
                
    def optimize(self, i):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(state_batch.shape)
        # print(action_batch.shape)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        # if i % self.print_ratio == 0:
        #     print(self.policy_net(state_batch))
        #     print(action_batch.unsqueeze(1))
        #     print(state_action_values)
        
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.cat([torch.from_numpy(state).float()], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.cat([torch.from_numpy(next_state).float()], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))
        
  
    def train(self):
        for i in trange(self.num_episodes):
            reward_in_episode = 0
            state, info = self.env.reset()
            state = state.transpose((2, 0, 1))
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                # print(action.item())
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                # observation = observation.reshape((192))
                observation = observation.transpose((2, 0, 1))

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                self.optimize(i)
                state = next_state
                reward_in_episode += reward

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if done or t > self.horizon:
                    self.reward_in_episode.append(reward_in_episode)
                    self.plot_rewards()
                    break
                
            # if i % self.target_update == 0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i % self.save_ratio == 0:
                # self._save()
                torch.save(self.target_net.state_dict(), f'{self.model_dir}/pytorch_{self.id}.pt')
                
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
