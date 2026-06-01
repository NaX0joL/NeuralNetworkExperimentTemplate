from dataclasses import dataclass

import torch
from torch import nn

from .master_config import MasterConfig
from .datasets.data_module import DataModule



@dataclass
class ExperimentContext:
    master_config: MasterConfig
    data_module: DataModule
    model: nn.Module
    loss_log: dict[str, list[float]]