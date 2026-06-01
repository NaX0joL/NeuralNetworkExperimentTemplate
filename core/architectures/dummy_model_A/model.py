import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DummyModelAConfig



class DummyModelA(nn.Module):
    def __init__(self, config:DummyModelAConfig):
        input_feature = config.input_feature
        hidden_size = config.hidden_size
        num_classes = config.num_classes
        
        super().__init__()
        
        self.fc1 = nn.Linear(input_feature, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        return
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x