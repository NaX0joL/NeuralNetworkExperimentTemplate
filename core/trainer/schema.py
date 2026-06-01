from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from core.datasets.data_module import DataModule

from .config import TrainerConfig
from .loss_manager import LossManager


@dataclass
class TrainerComponents:
    config: TrainerConfig
    data_module: DataModule
    model: nn.Module
    loss_log: dict[str, list]
    device: torch.device
    optimizer: optim.Optimizer
    criterion: nn.Module
    scheduler: lr_scheduler.LRScheduler
    loss_manager: LossManager

@dataclass
class TrainerState: 
    total_epochs: int = 0
    current_epoch: int = 0
    last_train_loss: float = 0.0
    last_validation_loss: float = 0.0