from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import wraps
import tqdm

import torch
from torch import optim, Tensor, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.abstract import ABSTRACT_Trainer
from core.datasets.data_module import DataModule

from .config import TrainerConfig
from .schema import TrainerComponents, TrainerState
from .loss.loss import LossManager
from .checkpoint_manager import CheckpointManager
from .progress_display import ProgressDisplay
from .loss_logger import LossLogger



@dataclass
class TrainerComponents:
    config: TrainerConfig
    data_module: DataModule
    model: nn.Module
    device: torch.device
    optimizer: optim.Optimizer
    scheduler: lr_scheduler.LRScheduler
    criterion: nn.Module
    loss_manager: LossManager



@dataclass
class TrainerState: 
    total_epochs: int = 0
    current_epoch: int = 0
    last_train_loss: float = 0.0
    last_validation_loss: float = 0.0
    train_loss_log: list = field(default_factory=list)
    validation_loss_log: list = field(default_factory=list)



class Trainer(ABSTRACT_Trainer):
    def __init__(self, config:TrainerConfig, model:nn.Module, data_module:DataModule) -> None:
        self.components = TrainerComponentsFactory(config, model, data_module).create()
        self.state = TrainerState()
        
        self.checkpoint_manager = CheckpointManager(self.components, self.state)
        self.loss_logger = LossLogger(self.state)
        self.progress_display = ProgressDisplay(self.components, self.state)
        
        self.state.total_epochs = self.components.config.train_epochs
        return
    
    ### --- public functions ---
    
    def fit(self, timer:bool=False) -> None:
        self.checkpoint_manager.reset()
        self.loss_logger.reset()
        self.progress_display.start()
        
        for epoch in range(self.components.config.train_epochs):
            self.state.current_epoch = epoch+1
            
            self._processOneEpoch(
                type='train', 
                dataloader=self.components.data_module.train_dataloader
            )
            self._processOneEpoch(
                type='validation', 
                dataloader=self.components.data_module.test_dataloader
            )
            
            self.checkpoint_manager.save()
            self.loss_logger.update()
            self.progress_display.epoch_update()
        
        self.checkpoint_manager.load_best()
        self.progress_display.finish()
        return
    
    def get_model(self) -> nn.Module:
        return self.components.model
    
    def get_loss_log(self) -> tuple[list, list]:
        return self.loss_logger.get_log()
    
    ### --- private helper functions ---
    
    def _processOneEpoch(self, type:str, dataloader:DataLoader) -> None:
        if type == "train":
            self.components.model.train()
            gradient_support = torch.enable_grad()
        
        elif type == "validation":
            self.components.model.eval()
            gradient_support = torch.no_grad()
            
        else:
            raise TypeError("trainer error! invalid epoch process type")
        
        total_loss = 0
        
        with gradient_support:
            for data_batch in dataloader:
                
                if type == "train":
                    self.components.optimizer.zero_grad()
                
                loss = self._getBatchLoss(data_batch)     
                total_loss += loss.item()
                
                if type == "train":
                    self.progress_display.batch_update()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.components.model.parameters(), 
                        max_norm=self.components.config.grad_clip_max_norm,
                    )
                    self.components.optimizer.step()
                    self.components.scheduler.step()
                
        total_loss /= len(dataloader)
        
        if type == "train":
            self.state.last_train_loss = total_loss
        elif type == "validation":
            self.state.last_validation_loss = total_loss
        
        return
    
    def _getBatchLoss(self, data_batch:dict[str, list|Tensor]) -> Tensor:
        input = data_batch.get("value").to(self.components.device)
        label = data_batch.get("ground_truth").to(self.components.device)
        
        output = self.components.model(input)
        loss = self.components.loss_manager.forward(output, label)
        return loss
    
    def _modifyDataBatch(self, data_batch:dict[str, list|Tensor]) -> dict[str, list|Tensor]:
        data_batch["value"] = data_batch.get('value').float().to(self.device)
        # data_batch["ground_truth"] = data_batch.get('ground_truth').squeeze(1).float().to(self.device)
        data_batch["ground_truth"] = data_batch.get('ground_truth').float().to(self.device)
        
        return data_batch



class TrainerComponentsFactory():
    def __init__(self, config:TrainerConfig, model:nn.Module, data_module:DataModule) -> None:
        self.config = config
        self.model = model
        self.data_module = data_module
        
        self._setDevice()
        self._setOptimizer()
        self._setCriterion()
        self._setScheduler()
        self._setLossManager()

        return 
    
    ### --- public functions ---
    
    def create(self) -> TrainerComponents:
        trainer_components = TrainerComponents(
            config=self.config,
            model=self.model,
            data_module=self.data_module,
            device=self.device,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            loss_manager=self.loss_manager,
        )
        return trainer_components
    
    ### --- private helper functions ---
    
    def _setDevice(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return
    
    def _setOptimizer(self) -> None:
        self.optimizer = self.config.optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return
        
    def _setCriterion(self) -> nn.Module:
        self.criterion = self.config.criterion_class()
        return
    
    def _setScheduler(self) -> None:
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer = self.optimizer,
            steps_per_epoch = len(self.data_module.train_dataloader),
            epochs = self.config.train_epochs,
            pct_start = self.config.percentage_start,
            max_lr = self.config.max_learning_rate,
            cycle_momentum = False if self.config.optimizer_name == 'adamw' else True,
        )
        return
    
    def _setLossManager(self) -> None:
        self.loss_manager = LossManager(self.config.loss_config, self.criterion)
        return