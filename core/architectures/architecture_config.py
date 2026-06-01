from dataclasses import dataclass

from core.abstract import ABSTRACT_Config

from .registry import ARCHITECTURE_REGISTRY



@dataclass
class ArchitectureConfig(ABSTRACT_Config):
    architecture_name: str
    arch_specific_config: ABSTRACT_Config = None

    def __post_init__(self):
        if self.arch_specific_config is None:
            entry = ARCHITECTURE_REGISTRY[self.architecture_name]
            self.arch_specific_config = entry.config_class.default()
        return

    @classmethod
    def default(self):
        name = "dummy_model_A"
        entry = ARCHITECTURE_REGISTRY[name]
        
        architecture_config = self(
            architecture_name = name,
            arch_specific_config = entry.config_class.default(),
        )
        return architecture_config