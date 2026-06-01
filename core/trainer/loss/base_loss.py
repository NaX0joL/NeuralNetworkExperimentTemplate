
from torch import nn, Tensor

from core.abstract import ABSTRACT_Loss



class Base_Loss(ABSTRACT_Loss, nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = None
        return
    
    ### public functions
    
    def set_criterion(self, criterion:nn.Module):
        self.criterion = criterion
        return
    
    def forward(self, x:Tensor, y:Tensor) -> Tensor:
        self._check_criterion()        
        loss = self.criterion(x, y)
        return loss
        
    ### private helper functions
    
    def _check_criterion(self):
        if self.criterion is None:
            raise AttributeError("Base Loss error! criterion not set")
        return