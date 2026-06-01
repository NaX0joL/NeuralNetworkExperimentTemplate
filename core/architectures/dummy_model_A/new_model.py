from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.abstract import ABSTRACT_Config



class DummyModelA(nn.Module):
    def __init__(self, config:DummyModelAConfig=None):
        if config is None:
            config = DummyModelAConfig.default()
        
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



@dataclass
class DummyModelAConfig(ABSTRACT_Config):
    input_feature: int
    hidden_size: int
    num_classes: int
    
    @classmethod
    def default(self):
        dummy_model_A_config = self(
            input_feature = 500,
            hidden_size = 10,
            num_classes = 2,
        )
        return dummy_model_A_config