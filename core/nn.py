import torch
import torch.nn as nn
import torch.nn.functional as F
from core.activations.fta import FTA


class Network(nn.Module):
    
    def __init__(self, use_fta):
        super(Network, self).__init__()
        self.use_fta = use_fta
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(8*8*16, 32)
        self.fta = FTA(tiles=20, bound_low=-2, bound_high=+2, eta=0.4, input_dim=32)
        if self.use_fta:
            self.q_network_fc1 = nn.Linear(640, 64)
        else:
            self.q_network_fc1 = nn.Linear(32, 64)

        self.q_network_fc2 = nn.Linear(64, 64)
        self.q_network_fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = x/255.0
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = x.reshape((-1, 1024))
        
        if self.use_fta:
            x = self.fta(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        
        print(x.shape)
        x = F.relu(self.q_network_fc1(x))
        x = F.relu(self.q_network_fc2(x))
        x = self.q_network_fc3(x)
        return x