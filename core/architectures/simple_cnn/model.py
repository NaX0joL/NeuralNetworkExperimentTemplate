import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F

from .config import SimpleCNNConfig



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



class Encoder(nn.Module):
    def __init__(self, in_channels:list[int], kernel_sizes:list[int], paddings:list[int]):
        super().__init__()
        
        if not (len(in_channels) == len(kernel_sizes) == len(paddings)):
            raise ValueError("Encoder error, invalid parameter size")
        
        conv_list = []
        for index in range(len(in_channels)):
            if index == 0:
                conv_list.append(
                    nn.Conv1d(1, in_channels[index], kernel_size=kernel_sizes[index], padding=paddings[index])
                )
            else:
                conv_list.append(
                    nn.Conv1d(in_channels[index-1], in_channels[index], kernel_size=kernel_sizes[index], padding=paddings[index])
                )
                
        self.convs = nn.ModuleList(conv_list)
        self.poolings = nn.ModuleList([nn.MaxPool1d(2, 2, return_indices=True) for _ in range(len(in_channels))])
        
        return
    
    def forward(self, x:Tensor) -> Tensor:
        pooling_indices = []
        tensor_sizes = []
        
        for conv, pooling in zip(self.convs, self.poolings):
            x = F.relu(conv(x))
            tensor_sizes.append(x.size())
            x, indices = pooling(x)
            pooling_indices.append(indices)
        
        return x, pooling_indices, tensor_sizes



class Latent(nn.Module):
    def __init__(self, n_layer:int, dim:int):
        super().__init__()
        
        self.n_layer = n_layer
        self.dim = dim
        
        self.connector_front = None
        self.flatten = nn.Flatten()
        
        latent_list = []
        for _ in range(n_layer):
            latent_list.append(nn.Linear(dim, dim))
            latent_list.append(nn.ReLU())
        self.latent = nn.ModuleList(latent_list)
        
        self.unflatten = None
        self.connector_back = None
        
        return
    
    def _firstForwardInit(self, x:Tensor) -> None:
        _, channels, seq_len = x.shape
        flattened_size = channels * seq_len
        
        self.connector_front = nn.Linear(flattened_size, self.dim).to(x.device)
        self.connector_back = nn.Linear(self.dim, flattened_size).to(x.device)
        self.unflatten = nn.Unflatten(1, (channels, seq_len)).to(x.device)
        
        return
    
    def forward(self, x:Tensor) -> Tensor:
        if self.unflatten is None:
            self._firstForwardInit(x)
        
        x = self.flatten(x)
        x = self.connector_front(x)
        for layer in self.latent:
            x = layer(x)
        x = self.connector_back(x)
        x = self.unflatten(x)
        
        return x



class Decoder(nn.Module):
    def __init__(self, in_channels:list[int], kernel_sizes:list[int], paddings:list[int]):
        super().__init__()
        
        if not (len(in_channels) == len(kernel_sizes) == len(paddings)):
            raise ValueError("Encoder error, invalid parameter size")
        
        deconv_list = []
        for index in range(len(in_channels)):
            if index < len(in_channels) - 1:
                deconv_list.append(
                    nn.ConvTranspose1d(in_channels[index], in_channels[index+1], kernel_size=kernel_sizes[index], padding=paddings[index])
                )
            else:
                deconv_list.append(
                    nn.ConvTranspose1d(in_channels[index], 1, kernel_size=kernel_sizes[index], padding=paddings[index])
                )
        
        self.deconvs = nn.ModuleList(deconv_list)
        self.unpools = nn.ModuleList([nn.MaxUnpool1d(2, 2) for _ in range(len(in_channels))])
        
        return

    def forward(self, x:Tensor, pooling_indices:list, tensor_sizes:list) -> Tensor:
        
        zipper = zip(self.unpools, self.deconvs, reversed(pooling_indices), reversed(tensor_sizes))
        for index, (unpool, deconv, indices, size) in enumerate(zipper):
            x = unpool(x, indices, output_size=size)
            x = deconv(x)
            if index < len(self.unpools) - 1:
                x = F.relu(x)
        
        return x