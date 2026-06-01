from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import optim, nn

from core.abstract import ABSTRACT_Config, ABSTRACT_Loss
from .loss.config import LossConfig



@dataclass
class TrainerConfig(ABSTRACT_Config):
    OPTIMIZER: ClassVar[dict[str, optim.Optimizer]] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }
    
    CRITERION: ClassVar[dict[str, nn.Module]] = {
        "MSE": nn.MSELoss,
        "MAE": nn.L1Loss,
        "hube ": nn.HuberLoss,
        "CrossEntropy": nn.CrossEntropyLoss,
    }
    
    train_epochs: int
    learning_rate: float
    max_learning_rate: float
    percentage_start: float
    weight_decay: float
    grad_clip_max_norm: float
    
    optimizer_name: str
    criterion_name: str
    
    use_best_model: bool
    checkpoint_path: str
    
    loss_config: LossConfig
    
    def __post_init__(self):
        self.optimizer_class = self.OPTIMIZER[self.optimizer_name]
        self.criterion_class = self.CRITERION[self.criterion_name]
        return
    
    @classmethod
    def default(self):
        loss_config = LossConfig.default()
        trainer_config = self(
            train_epochs = 100,
            learning_rate = 1e-4,
            max_learning_rate = 1e-2,
            percentage_start = 0.3,
            weight_decay = 1e-4,
            grad_clip_max_norm = 1.0,
            
            optimizer_name = 'adamw',
            criterion_name = 'MSE',
            
            use_best_model = False,
            checkpoint_path = 'core/savefolder/checkpoint',
            
            loss_config = loss_config,
        )
        return trainer_config