from collections import namedtuple, deque
import random
import numpy as np
import torch
import gymnasium as gym

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
def generate_random_trajectories(env:gym.Env, n=1000):
    buffer = ReplayMemory(capacity=1000000)
    for i in range(n):
        state, info = env.reset()
        done = False
        t = 0
        while not done:
            state = state.transpose((2, 0, 1))
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, action, observation, reward, None)
            t+=1
            done = terminated or truncated or t>100
            
            state = observation
            
    return buffer.sample(batch_size=n)

def create_distance_matrices(env, model, n=1000):
    random_transitions = generate_random_trajectories(env=env)
    # print(random_transitions[0].state)
    d_v = np.zeros((n, n))
    d_s = np.zeros((n, n))
    
    for i in range (n):
        print(i)
        temp_i = model.target_net(torch.tensor(random_transitions[i].state, dtype=torch.float32, device=model.device).unsqueeze(0))
        for j in range(i+1):
            temp_j = model.target_net(torch.tensor(random_transitions[j].state, dtype=torch.float32, device=model.device).unsqueeze(0))
            d_v[i, j] = abs(temp_i[0].cpu().detach().numpy() - temp_j[0].cpu().detach().numpy()).max()
            a = np.array((temp_i[2].cpu().detach().numpy() - temp_j[2].cpu().detach().numpy()))
            d_s[i, j] = np.linalg.norm(a)
            
            d_v[j, i] = d_v[i, j]
            d_s[j, i] = d_s[i, j]    
    
    return d_v, d_s

def create_representations(env, model):
    random_transitions = generate_random_trajectories(env=env)

    reps = []
    for i in range(1000):
        temp = model.target_net(torch.tensor(random_transitions[i].state, dtype=torch.float32, device=model.device).unsqueeze(0))
        rep = temp[2].cpu().detach().numpy()
        
        reps.append(rep)
    
    return np.array(reps)