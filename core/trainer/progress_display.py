from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from tqdm import tqdm, trange

from .config import TrainerConfig
from .schema import TrainerComponents, TrainerState



class ProgressDisplay():
    def __init__(self, trainer_components:TrainerComponents):
        self.trainer_components = trainer_components
        
        self.epoch_pbar = None
        return
    
    def start(self):
        print("Model training started!")
        self.epoch_pbar = tqdm(
            total=self.trainer_components.config.train_epochs,
            desc="progress",
            position=0,
            leave=True,
        )
        return
    
    def epoch_update(self, epoch:int):
        self.epoch_pbar.update(1)
        self.epoch_pbar.set_postfix({
            "epoch": f"{epoch}/{self.trainer_components.config.train_epochs}",
            "train_loss": f"{self.trainer_components.loss_log["train"][-1]:.5f}",
            "validation_loss": f"{self.trainer_components.loss_log["validation"][-1]:.5f}",
        })
        return
    
    def batch_update(self):
        return
    
    def finish(self):
        if self.epoch_pbar:
            self.epoch_pbar.close()
        print("Model training finished!")
        return