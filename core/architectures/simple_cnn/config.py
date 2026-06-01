from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class SimpleCNNConfig(ABSTRACT_Config):
    in_channels: list[int]
    kernel_sizes: list[int]
    paddings: list[int]
    
    linear_n_layer: int
    linear_dim: int
    
    @classmethod
    def default(self):
        simple_cnn_config = self(
            in_channels = [64, 128, 256, 512],
            kernel_sizes = [3, 3, 3, 3],
            paddings = [1, 1, 1, 1],
            
            linear_n_layer = 1,
            linear_dim = 1000,
        )
        return simple_cnn_config