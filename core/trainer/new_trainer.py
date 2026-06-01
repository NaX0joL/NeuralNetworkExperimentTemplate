from __future__ import annotations
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.abstract import ABSTRACT_Trainer
from core.schema import ExperimentContext
from core.modules.timer import use_timer 

from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
from .schema import TrainerComponents, TrainerState
from .epoch_processor import EpochProcessor
from .loss_manager import LossManager
from .checkpoint_manager import CheckpointManager
from .progress_display import ProgressDisplay
from .trainer_config import TrainerConfig
from ..datasets.data_module import DataModule



# @dataclass
# class TrainerComponents:
#     config: TrainerConfig
#     data_module: DataModule
#     model: nn.Module
#     loss_log: dict[str, list]
#     device: torch.device
#     optimizer: optim.Optimizer
#     criterion: nn.Module
#     scheduler: lr_scheduler.LRScheduler
#     loss_manager: LossManager

# @dataclass
# class TrainerState: 
#     total_epochs: int = 0
#     current_epoch: int = 0
#     last_train_loss: float = 0.0
#     last_validation_loss: float = 0.0
#     train_loss_log: list = field(default_factory=list)
#     validation_loss_log: list = field(default_factory=list)



class Trainer(ABSTRACT_Trainer):
    def __init__(self, experiment_context:ExperimentContext):
        self.experiment_context = experiment_context
        
        self.trainer_components = self.trainer_components = TrainerComponentsFactory(
            config = experiment_context.master_config.trainer_config, 
            data_module = experiment_context.data_module,
            model = experiment_context.model,
            loss_log = experiment_context.loss_log,
        ).create()

        self.epoch_processor = EpochProcessor(self.trainer_components)
        self.checkpoint_manager = CheckpointManager(self.trainer_components)
        self.progress_display = ProgressDisplay(self.trainer_components)
        return
    
    ### --- public functions ---
    
    def get_model(self) -> nn.Module:
        return self.trainer_components.model
    
    def get_loss_log(self) -> dict[str, list]:
        return self.trainer_components.loss_log
    
    @use_timer
    def fit(self, timer=False) -> None:
        self._empty_log()
        
        self.checkpoint_manager.reset()
        self.progress_display.start()
        
        for epoch in range(self.trainer_components.config.train_epochs):
            
            self.epoch_processor.process_epoch(
                epoch_type='train', 
                dataloader=self.trainer_components.data_module.train_dataloader
            )
            self.epoch_processor.process_epoch(
                epoch_type='validation', 
                dataloader=self.trainer_components.data_module.test_dataloader
            )
            
            self.checkpoint_manager.save(epoch+1)
            self.progress_display.epoch_update(epoch+1)
        
        self.checkpoint_manager.load_best()
        self.progress_display.finish()
        return
    
    ### --- private helper functions ---
    
    def _empty_log(self) -> None:
        self.trainer_components.loss_log["train"].clear()
        self.trainer_components.loss_log["validation"].clear()
        return



class TrainerComponentsFactory():
    def __init__(self, config:TrainerConfig, model:nn.Module, data_module:DataModule, loss_log:dict[str, list]) -> None:
        self.config = config
        self.data_module = data_module
        self.model = model
        self.loss_log = loss_log
        
        self._resolve_device()
        self._set_optimizer()
        self._set_criterion()
        self._set_scheduler()
        self._set_loss_manager()
        return
    
    ### --- public functions ---
    
    def create(self) -> TrainerComponents:
        trainer_components = TrainerComponents(
            config = self.config,
            data_module = self.data_module,
            model = self.model,
            loss_log = self.loss_log,
            device = self.device,
            optimizer = self.optimizer,
            criterion = self.criterion,
            scheduler = self.scheduler,
            loss_manager = self.loss_manager,
        )
        return trainer_components
    
    ### --- private helper functions ---
    
    def _resolve_device(self) -> None:
        self.device = next(self.model.parameters()).device
        return
    
    def _set_optimizer(self) -> None:
        self.optimizer: optim.Optimizer = OPTIMIZER_REGISTRY[self.config.optimizer_name](
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # to suppress warning sign
        self.optimizer.step()
        self.optimizer.zero_grad()
        return 
    
    def _set_criterion(self) -> None:
        self.criterion = CRITERION_REGISTRY[self.config.criterion_name]()
        return
    
    def _set_scheduler(self) -> None:
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer = self.optimizer,
            steps_per_epoch = len(self.data_module.train_dataloader),
            epochs = self.config.train_epochs,
            pct_start = self.config.percentage_start,
            max_lr = self.config.max_learning_rate,
            cycle_momentum = False if self.config.optimizer_name == "adamw" else True,
        )
        return
    
    def _set_loss_manager(self) -> None:
        self.loss_manager = LossManager(
            criterion = self.criterion,
            loss_coefficients = {
                "base_loss": 1,
            },
        )
        return