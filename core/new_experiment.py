from dataclasses import dataclass, field

import os
import random
import numpy as np
from pathlib import Path

import torch
from torch import Tensor, nn

from .schema import ExperimentContext
from .master_config import MasterConfig
from .architectures.new_architecture_control import ArchitectureControl
from .datasets.new_data_module import DataModuleGetter, DataModule
from .trainer.new_trainer import Trainer
from .saving.mpkg import MPKG
from .metrics.metrics import MetricsEvaluator



# @dataclass
# class ExperimentContext:
#     master_config: MasterConfig
#     data_module: DataModule
#     model: nn.Module
#     loss_log: dict[str, list[float]]



class Experiment():
    
    def __init__(self, master_config:MasterConfig=None, experiment_id:str="tmp_exp", random_seed:int=42, determinism:bool=True) -> None:
        self.master_config = master_config
        self.experiment_id = experiment_id
        self.random_seed = random_seed
        self.determinism = determinism
        
        self._set_random_seed()
        self._set_pytorch_determinism()
        self._reload_experiment_context()
        return
        
    ### public functions
    
    def save_to_mpkg(self, path:Path=None):
        mpkg = MPKG(self.experiment_context)
        mpkg.add_extra(
            name = self.experiment_id,
            seed = self.random_seed,
            determinism = self.determinism,
        )
        mpkg.save_to_mpkg(path)
        pass
    
    def load_from_mpkg(self):
        # load context from mpkg
        # re-create context
        pass
    
    def get_model_metrics(self):
        metrics_evaluator = MetricsEvaluator(self.experiment_context)
        metrics_result = metrics_evaluator.calculate()
        return metrics_result
    
    def train_model(self, timer=False) -> None:
        trainer = Trainer(self.experiment_context)
        trainer.fit(timer=timer)
        return
    
    def model_inference(self, input:Tensor) -> Tensor:
        self.experiment_context.model.eval()
        with torch.no_grad():
            input = input.float().to(self.experiment_context.device)
            prediction = self.experiment_context.model(input)
        return prediction
    
    ### private helper functions
    
    def _set_random_seed(self) -> None:
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        return
    
    def _set_pytorch_determinism(self) -> None:
        if self.determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return
    
    def _reload_experiment_context(self) -> None:
        self.experiment_context = ExperimentContextFactory(
            self.master_config
        ).create()
        return



class ExperimentContextFactory():
    def __init__(self, master_config:MasterConfig) -> None:
        self.master_config = master_config
        
        self._set_device()
        self._set_data_module()
        self._set_model()
        self._set_loss_log()
        return
    
    ### public functions
    
    def create(self) -> ExperimentContext:
        experiment_components = ExperimentContext(
            master_config = self.master_config,
            data_module = self.data_module,
            model = self.model,
            loss_log = self.loss_log,
        )
        return experiment_components
    
    ### private helper functions
    
    def _set_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return
    
    def _set_data_module(self) -> None:
        self.data_module = DataModuleGetter(
            self.master_config.dataset_config
        ).get_data_module()
        return
    
    def _set_model(self) -> None:
        self.model = ArchitectureControl(
            self.master_config.architecture_config
        ).get_model().to(self.device)
        return
    
    def _set_loss_log(self) -> None:
        self.loss_log = {"train": [], "validation": []}
        return