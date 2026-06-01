from __future__ import annotations 
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from core.abstract import ABSTRACT_Config

from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
from .loss_manager import LossManager
from ..datasets.data_module import DataModule



@dataclass
class TrainerConfig(ABSTRACT_Config):
    train_epochs: int
    
    learning_rate: float
    max_learning_rate: float
    percentage_start: float
    
    weight_decay: float
    grad_clip_max_norm: float
    
    optimizer_name: str
    criterion_name: str
    
    use_best_model: bool
    
    @classmethod
    def default(self):
        trainer_config = self(
            train_epochs = 100,
            
            learning_rate = 1e-4,
            max_learning_rate = 1e-3,
            percentage_start = 0.3,
            
            weight_decay = 1e-4,
            grad_clip_max_norm = 1.0,
            
            optimizer_name = 'adamw',
            criterion_name = 'MSE',
            
            use_best_model = True,
            #checkpoint_path = 'core/savefolder/checkpoint',
        )
        return trainer_config