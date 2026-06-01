from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class UNetConfig(ABSTRACT_Config):
    n_channels: int
    n_classes: int
    bilinear: bool
        
    down_kernel_size: int
    up_kernel_size: int
    out_kernel_size: int
    
    @classmethod
    def default(self):
        unet_config = self(
            n_channels = 1,
            n_classes = 1,
            bilinear = False,
                
            down_kernel_size = 3,
            up_kernel_size = 2,
            out_kernel_size = 1,
        )
        return unet_config