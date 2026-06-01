import torch
from torch import nn
from torch import Tensor

from .registry import LOSS_REGISTRY



class LossManager(nn.Module):
    def __init__(self, criterion:nn.Module, loss_coefficients:dict[str, float]):
        super().__init__()
        self.criterion = criterion
        self._set_used_loss(loss_coefficients)
        return
    
    ### public functions
    
    def forward(self, x:Tensor, y:Tensor):
        total_loss = 0

        for loss_inst, coeff in self.used_loss:
            loss = loss_inst(x, y)
            total_loss += loss * coeff
        
        return total_loss
    
    ### private helper functions
    
    def _set_used_loss(self, loss_coefficients:dict[str, float]) -> None:
        self.used_loss: list[tuple[nn.Module, float]] = []
        for loss_name, loss_class in LOSS_REGISTRY.items():
            if loss_name in loss_coefficients.keys():
                loss_inst = loss_class()
                loss_inst.set_criterion(self.criterion)
                
                self.used_loss.append((loss_inst, loss_coefficients[loss_name]))
        return