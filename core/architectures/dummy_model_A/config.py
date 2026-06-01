from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class DummyModelAConfig(ABSTRACT_Config):
    input_feature: int
    hidden_size: int
    num_classes: int
    
    @classmethod
    def default(self):
        dummy_model_A_config = self(
            input_feature = 500,
            hidden_size = 10,
            num_classes = 2,
        )
        return dummy_model_A_config