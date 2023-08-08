import torch
import torch.nn as nn
import torch.nn.functional as F
from core.activations.fta import FTA

class SFNetwork(nn.Module):
    def __init__(self, use_fta, action_dim=1):
        super().__init__()
        self.use_fta = use_fta
        
        if self.use_fta:
            self.linear1 = nn.Linear(640+action_dim, 640)
            self.linear2 = nn.Linear(640, 640)
        else:
            self.linear1 = nn.Linear(32+action_dim, 32)
            self.linear2 = nn.Linear(32, 32)
        
    def forward(self, x, actions):
        x = torch.cat((x, actions), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
        
        

class Reward(nn.Module):
    def __init__(self, use_fta, action_dim=1):
        super().__init__()
        self.use_fta = use_fta
        if self.use_fta:
            self.linear1 = nn.Linear(640+action_dim, 1024)
        else:
            self.linear1 = nn.Linear(32+action_dim, 1024)
        
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 1)
        
    def forward(self, x, actions):
        x = torch.cat((x, actions), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x
        
class InputReconstruction(nn.Module):
    def __init__(self, use_fta):
        super(InputReconstruction, self).__init__()
        if use_fta:
            self.linear = nn.Linear(640, 1024)
        else:
            self.linear = nn.Linear(32, 1024)
        self.unflat = nn.Unflatten(1, (16, 8, 8))
        self.convT1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=2)
        self.convT2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.unflat(x)
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        
        return x
    
class VirtualValueFunction(nn.Module):
    def __init__(self, use_fta):
        super(VirtualValueFunction, self).__init__()
        self.use_fta = use_fta

        if self.use_fta:
            self.q_network_fc1 = nn.Linear(640, 64)
        else:
            self.q_network_fc1 = nn.Linear(32, 64)
            
        self.q_network_fc2 = nn.Linear(64, 64)
        self.q_network_fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        
        x = F.relu(self.q_network_fc1(x))
        x = F.relu(self.q_network_fc2(x))
        x = self.q_network_fc3(x)
        
        return x
        
class Network(nn.Module):
    
    def __init__(self, use_fta, use_aux=None):
        super(Network, self).__init__()
        self.use_fta = use_fta
        self.use_aux = use_aux
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*16, 32)
        self.fta = FTA(tiles=20, bound_low=-2, bound_high=+2, eta=0.4, input_dim=32)
        if self.use_fta:
            self.q_network_fc1 = nn.Linear(640, 64)
        else:
            self.q_network_fc1 = nn.Linear(32, 64)

        if self.use_aux != "no_aux":
            if self.use_aux == 'ir':
                self.aux_network = InputReconstruction(use_fta=self.use_fta)
            elif self.use_aux == 'reward':
                self.aux_network = Reward(use_fta=self.use_fta)
            elif self.use_aux == 'sf':
                self.aux_network = SFNetwork(use_fta=self.use_fta)
            elif self.use_aux == 'sf+reward':
                self.aux_network = SFNetwork(use_fta=self.use_fta)
                self.reward_network = Reward(use_fta=self.use_fta)
            elif self.use_aux == 'virtual-reward-1' or self.use_aux == 'virtual-reward-5':
                self.aux_network = VirtualValueFunction(use_fta=self.use_fta)
        
        self.q_network_fc2 = nn.Linear(64, 64)
        self.q_network_fc3 = nn.Linear(64, 4)
        
    def forward(self, x, actions=None):
        x = x/255.0
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = x.reshape((-1, 1024))
        
        if self.use_fta:
            rep = self.fta(self.fc1(x))
        else:
            rep = F.relu(self.fc1(x))
        # representation built
        # auxilary network
        aux = None
        reward = None
        
        if self.use_aux != "no_aux":
            if self.use_aux == "reward":
                if actions != None:
                    aux = self.aux_network(rep, actions)
            elif self.use_aux == "sf":
                if actions != None:
                    aux = self.aux_network(rep, actions)
            elif self.use_aux == "sf+reward":
                if actions != None:
                    aux = self.aux_network(rep, actions)
                    reward = self.reward_network(rep, actions)
            elif self.use_aux == "ir":
                aux = self.aux_network(rep)
            elif self.use_aux == "virtual-reward-1" or self.use_aux == "virtual-reward-5":
                aux = self.aux_network(rep)
            else:
                aux=None    
        # value network
        x = F.relu(self.q_network_fc1(rep))
        x = F.relu(self.q_network_fc2(x))
        x = self.q_network_fc3(x)

        return [x, aux, rep, reward]