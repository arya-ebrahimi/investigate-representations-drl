import torch
import torch.nn as nn
import torch.nn.functional as F
from core.activations.fta import FTA

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.fc1 = nn.Linear(8*8*16, 32)
        
        self.fta = FTA(tiles=20, bound_low=-2, bound_high=+2, eta=0.4, input_dim=32)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x)
        x = self.fta(self.fc1(x))
        
        return x