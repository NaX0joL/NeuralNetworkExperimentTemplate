from dataclasses import dataclass

from torch import nn

from core.abstract import ABSTRACT_Config

from .dummy_model_A.model import DummyModelA, DummyModelAConfig
from .proposed_model.model import ProposedModel, ProposedModelConfig
from .simple_cnn.model import SimpleCNN, SimpleCNNConfig
from .traditional_matrix_profile.model import TraditionalMatrixProfile, TraditionalMatrixProfileConfig
from .UNet.model import UNet, UNetConfig



@dataclass
class ArchitectureEntry:
    model_class:  type[nn.Module]
    config_class: type[ABSTRACT_Config]



ARCHITECTURE_REGISTRY: dict[str, ArchitectureEntry] = {
    "dummy_model_A": ArchitectureEntry(DummyModelA, DummyModelAConfig),
    "proposed_model": ArchitectureEntry(ProposedModel, ProposedModelConfig),
    "simple_cnn": ArchitectureEntry(SimpleCNN, SimpleCNNConfig),
    "traditional_matrix_profile": ArchitectureEntry(TraditionalMatrixProfile, TraditionalMatrixProfileConfig),
    "UNet": ArchitectureEntry(UNet, UNetConfig),
}