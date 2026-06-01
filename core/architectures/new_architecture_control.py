from __future__ import annotations
from dataclasses import dataclass

from torch import nn

from core.abstract import ABSTRACT_ArchitectureControl, ABSTRACT_Config

from .registry import ARCHITECTURE_REGISTRY
from .architecture_config import ArchitectureConfig



class ArchitectureControl(ABSTRACT_ArchitectureControl):
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config
        return

    def get_model(self) -> nn.Module:
        entry = ARCHITECTURE_REGISTRY[self.architecture_config.architecture_name]
        model = entry.model_class(self.architecture_config.arch_specific_config)
        return model