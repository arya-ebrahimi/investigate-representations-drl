import torch
import torch.nn as nn
import torch.nn.functional as F
from core.activations.fta import FTA

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, )