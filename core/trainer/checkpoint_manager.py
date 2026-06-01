from __future__ import annotations

import os
from dataclasses import dataclass
from functools import wraps

import torch
from torch import optim, nn

from .config import TrainerConfig
from .schema import TrainerComponents, TrainerState



CHECKPOINT_PATH = "core/savefolder/checkpoint"



def check_if_enabled(func:function):
    @wraps(func)
    def wrapper(self:CheckpointManager, *args, **kwargs):
        if not self.trainer_components.config.use_best_model:
            return
        return func(self, *args, **kwargs)
    return wrapper



class CheckpointManager():
    
    def __init__(self, trainer_components:TrainerComponents):
        self.trainer_components = trainer_components

        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        return
    
    ### --- public functions ---
    
    @check_if_enabled
    def reset(self) -> None:
        self.best_validation_loss = float('inf')
        self.best_epoch = -1
        return
    
    @check_if_enabled
    def save(self, epoch:int) -> None:
        last_validation_loss = self.trainer_components.loss_log["validation"][-1]
        
        if last_validation_loss < self.best_validation_loss:
            self.best_val_loss = last_validation_loss
            self.best_epoch = epoch
            
            checkpoint = {
                "best_val_loss": self.best_val_loss,
                "epoch": self.best_val_loss,
                "model_state_dict": self.trainer_components.model.state_dict(),
                "optimizer_state_dict": self.trainer_components.optimizer.state_dict(),
            }
            save_path = os.path.join(CHECKPOINT_PATH, "checkpoint.pth")
            torch.save(checkpoint, save_path)
            
        return
    
    @check_if_enabled
    def load_best(self) -> None:
        load_path = os.path.join(CHECKPOINT_PATH, "checkpoint.pth")
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.trainer_components.device)
            self.trainer_components.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise FileNotFoundError("trainer error! model checkpoint not found")
        
        return