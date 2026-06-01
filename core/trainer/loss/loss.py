from typing import Type

import torch 
from torch import Tensor, nn

from core.abstract import ABSTRACT_Loss

from .config import LossConfig



class LossManager():
    def __init__(self, config:LossConfig, criterion:nn.Module):
        self.LOSS: dict[str, Type[ABSTRACT_Loss]] = {
            "base_loss": BaseLoss,
            "matrix_profile_loss": MatrixProfileLoss,
        }
        
        self.config = config
        self.criterion = criterion
        return
    
    def forward(self, input, label) -> Tensor:
        total_loss = 0
        
        if self.config.use_multi_loss:
            for name, loss_class in self.LOSS.items():
                
                if self.config.multi_loss_coeff[name] == 0:
                    continue
                
                loss = loss_class(self.criterion).forward(input, label)
                total_loss += self.config.multi_loss_coeff[name] * loss
        
        else:
            # print(f"input shape, label shape: {input.shape}, {label.shape}")
            total_loss = self.criterion(input, label)
        
        return total_loss



class BaseLoss(ABSTRACT_Loss):
    def __init__(self, criterion):
        self.criterion = criterion
        return
    
    def forward(self, X, y) -> Tensor:
        pass



class MatrixProfileLoss(ABSTRACT_Loss):
    def __init__(self, criterion):
        self.criterion = criterion
        return
    
    def forward(self, X, y) -> Tensor:
        pass