import torch
from torch import Tensor, nn
import torch.nn.functional as F



class DoubleConv_1d(nn.Module):
    """_summary_
        (convolution => [BN] => ReLu) * 2
        but for 1d input
    """
    
    def __init__(self, in_channels:int, out_channels:int, mid_channels:int=None, kernel_size:int=3):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        return
    
    def forward(self, x:Tensor) -> Tensor:
        return self.double_conv(x)



class Down_1d(nn.Module):
    """_summary_
        Downscaling with maxpool then double conv
        but for 1d input
    """
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv_1d(in_channels, out_channels, kernel_size=kernel_size)
        )
        
        return

    def forward(self, x:Tensor) -> Tensor:
        return self.maxpool_conv(x)



class Up_1d(nn.Module):
    """_summary_
        Upscaling then double conv
        but for 1d input
    """
    
    def __init__(self, in_channels:int, out_channels:int, linear:bool=True, kernel_size:int=2):
        super().__init__()

        # if linear, use the normal convolutions to reduce the number of channels
        if linear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='linear', align_corners=True)
            self.conv = DoubleConv_1d(in_channels, out_channels, in_channels // kernel_size, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // kernel_size, kernel_size=kernel_size, stride=kernel_size)
            self.conv = DoubleConv_1d(in_channels, out_channels, kernel_size=kernel_size)
        
        return

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        x1 = self.up(x1)
        
        # x1: [batch, channels, seq_len1]
        # x2: [batch, channels, seq_len2] where seq_len2 > seq_len1
        diff = x2.size()[2] - x1.size()[2]

        # pad tensors so both have dim
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        # combine both tensors
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)



class OutConv_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv_1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        return
    
    def forward(self, x:Tensor) -> Tensor:
        return self.conv(x)