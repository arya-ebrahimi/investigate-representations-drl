import torch
import torch.nn as nn
import torch.nn.functional as F
from core.activations.fta import FTA
from collections import namedtuple, deque
import numpy as np
import os
from pathlib import Path
import random
from tqdm import trange
from itertools import count
import matplotlib.pyplot as plt
import matplotlib
import math
import datetime


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = np.random.default_rng()

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.choice(np.arange(len(self.memory)), batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.memory[i])
        return res

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(4*4*16, 32)
        self.fta = FTA(tiles=20, bound_low=-2, bound_high=+2, eta=0.4, input_dim=32)
        
        self.q_network_fc1 = nn.Linear(32, 32)
        self.q_network_fc2 = nn.Linear(32, 32)
        self.q_network_fc3 = nn.Linear(32, 4)
        
    def forward(self, x):
        x = x/255.0
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        
        x = F.relu(self.fc1(x.reshape(-1, 256)))
                
        x = F.relu(self.q_network_fc1(x))
        x = F.relu(self.q_network_fc2(x))
        x = self.q_network_fc3(x)
        return x

class Agent():
    def __init__(self, env):
        self.env = env
        self.num_episodes = 75000
        self.save_ratio=1000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 256
        self.gamma = 0.99
        self.eps_start = 1
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.target_update = 1000
        self.learning_rate = 0.000025
        self.max_episode = 50
        self.id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.model_dir = Path('.models')
        self.tau = 0.005

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.action_space = env.action_space.n
        
        self.policy_net = Network().to(self.device)
        self.target_net = Network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        
        self.memory = ReplayMemory(1000000)
        
        self.steps_done = 0
        self.reward_in_episode = []
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        # print(eps_threshold)
        self.steps_done += 1
        if sample < eps_threshold:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state.transpose((2, 0, 1)), device=self.device)).max(1)[1].item()

            
            
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
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
                
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_values = self.target_net(next_state_batch).max(1)[0]
        
        expected_action_values = (~done_batch * next_values * self.gamma) + reward_batch
        
        loss = self.loss_fn(action_values, expected_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))
        
  
    def train(self):
        for i in trange(self.num_episodes):
            
            state, _ = self.env.reset()
            done = False
            reward_in_episode = 0
            for t in count():
                action = self.select_action(state=state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self._remember(state.transpose((2, 0, 1)), action, next_state.transpose((2, 0, 1)), reward, done)
                
                self.optimize()
                state = next_state
                reward_in_episode += reward

                # target_net_state_dict = self.target_net.state_dict()
                # policy_net_state_dict = self.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                # self.target_net.load_state_dict(target_net_state_dict)
                
                if done or t > self.max_episode:
                    self.reward_in_episode.append(reward_in_episode)
                    self.plot_rewards()
                    break
                
            if i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i % self.save_ratio == 0:
                # self._save()
                torch.save(self.target_net.state_dict(), f'{self.model_dir}/pytorch_{self.id}.pt')
                
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
