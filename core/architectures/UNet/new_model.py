from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from core.abstract import ABSTRACT_Config

from .components import DoubleConv_1d, Down_1d, Up_1d, OutConv_1d



class UNet(nn.Module):
    def __init__(self, config:UNetConfig):
        super().__init__()
        
        n_channels = config.n_channels
        n_classes = config.n_classes
        bilinear = config.bilinear
        
        down_kernel_size = config.down_kernel_size
        up_kernel_size = config.up_kernel_size
        out_kernel_size = config.out_kernel_size
        
        # enc
        self.inc = DoubleConv_1d(n_channels, 64, kernel_size=down_kernel_size)
        self.down1 = Down_1d(64, 128, kernel_size=down_kernel_size)
        self.down2 = Down_1d(128, 256, kernel_size=down_kernel_size)
        self.down3 = Down_1d(256, 512, kernel_size=down_kernel_size)
        
        # idk what this factor is
        factor = 2 if bilinear else 1
        self.down4 = Down_1d(512, 1024 // factor, kernel_size=down_kernel_size)
        
        # dec
        self.up1 = Up_1d(1024, 512 // factor, bilinear, kernel_size=up_kernel_size)
        self.up2 = Up_1d(512, 256 // factor, bilinear, kernel_size=up_kernel_size)
        self.up3 = Up_1d(256, 128 // factor, bilinear, kernel_size=up_kernel_size)
        self.up4 = Up_1d(128, 64, bilinear, kernel_size=up_kernel_size)
        self.outc = OutConv_1d(64, n_classes, kernel_size=out_kernel_size)
        
        return
        
    def forward(self, x:Tensor) -> Tensor:
        # adjust for custom input shape
        x = x.permute(0, 2, 1)      # [bs, feature, seq_len]
        
        x1 = self.inc(x)            # [bs, 64, seq_len]
        x2 = self.down1(x1)         # [bs, 128, seq_len / 2]
        x3 = self.down2(x2)         # [bs, 256, seq_len / 4]
        x4 = self.down3(x3)         # [bs, 512, seq_len / 8]
        x5 = self.down4(x4)         # [bs, 1024, seq_len / 16]
        
        x = self.up1(x5, x4)        # [bs, 512, seq_len / 8]
        x = self.up2(x, x3)         # [bs, 256, seq_len / 4]
        x = self.up3(x, x2)         # [bs, 128, seq_len / 2]
        x = self.up4(x, x1)         # [bs, 64, seq_len]
        
        x = self.outc(x)            # [bs, feature, seq_len]
        
        x = x.permute(0, 2, 1)      # [bs, seq_len, feature]
        return x



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