import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import UNetConfig



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