from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from core.abstract import ABSTRACT_Config

from .components import Encoder, Latent, Decoder



class SimpleCNN(nn.Module):
    def __init__(self, config:SimpleCNNConfig):
        super().__init__()
        
        self.encoder = Encoder(config.in_channels, config.kernel_sizes, config.paddings)
        self.latent = Latent(config.linear_n_layer, config.linear_dim)
        self.decoder = Decoder(list(reversed(config.in_channels)), list(reversed(config.kernel_sizes)), list(reversed(config.paddings)))
        
        return
    
    def forward(self, x:Tensor) -> Tensor:
        # quick dimension addition for channel/feature
        x = x.unsqueeze(-1)
        
        x = x.permute(0, 2, 1)      # [bs, feature, seq_len]
        
        x, pooling_indices, tensor_sizes = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x, pooling_indices, tensor_sizes)
        
        x = x.permute(0, 2, 1)      # [bs, seq_len, feature]
        
        # quick dimension removal for channel/feature
        x = x.squeeze(-1)
        return x



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