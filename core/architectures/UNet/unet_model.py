""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        
        self.args = args
        
        self.dimension = args.unet_dimension
        self.n_channels = args.unet_n_channels
        self.n_classes = args.unet_n_classes
        self.bilinear = args.unet_bilinear
        
        self.down_kernel_size = args.unet_down_kernel
        self.up_kernel_size = args.unet_up_kernel
        self.out_kernel_size = args.unet_out_kernel

        # original
        if self.dimension == 2:
            self.inc = (DoubleConv(self.n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if self.bilinear else 1
            self.down4 = (Down(512, 1024 // factor))
            self.up1 = (Up(1024, 512 // factor, self.bilinear))
            self.up2 = (Up(512, 256 // factor, self.bilinear))
            self.up3 = (Up(256, 128 // factor, self.bilinear))
            self.up4 = (Up(128, 64, self.bilinear))
            self.outc = (OutConv(64, self.n_classes))
        
        # modified
        elif self.dimension == 1:
            # enc                                                                                       # [bs, feature, seq_len]
            self.inc = (DoubleConv_1d(self.n_channels, 64, kernel_size=self.down_kernel_size))          # [bs, 64, seq_len]
            self.down1 = (Down_1d(64, 128, kernel_size=self.down_kernel_size))                          # [bs, 128, seq_len / 2]
            self.down2 = (Down_1d(128, 256, kernel_size=self.down_kernel_size))                         # [bs, 256, seq_len / 4]
            self.down3 = (Down_1d(256, 512, kernel_size=self.down_kernel_size))                         # [bs, 512, seq_len / 8]
            
            # idk what this factor is
            factor = 2 if self.bilinear else 1
            self.down4 = (Down_1d(512, 1024 // factor, kernel_size=self.down_kernel_size))              # [bs, 1024, seq_len / 16]
            
            # dec 
            self.up1 = (Up_1d(1024, 512 // factor, self.bilinear, kernel_size=self.up_kernel_size))     # [bs, 512, seq_len / 8]
            self.up2 = (Up_1d(512, 256 // factor, self.bilinear, kernel_size=self.up_kernel_size))      # [bs, 256, seq_len / 4]
            self.up3 = (Up_1d(256, 128 // factor, self.bilinear, kernel_size=self.up_kernel_size))      # [bs, 128, seq_len / 2]
            self.up4 = (Up_1d(128, 64, self.bilinear, kernel_size=self.up_kernel_size))                 # [bs, 64, seq_len]
            self.outc = (OutConv_1d(64, self.n_classes, kernel_size=self.out_kernel_size))              # [bs, feature, seq_len]
        
        # raise error when dimension is not valid
        else:
            assert True, "dimension of UNet must be specified"

    def forward(self, x):
        # adjust for custom input shape
        x = x.permute(0, 2, 1)              # [bs, channel, seq_len]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        logits = logits.permute(0, 2, 1)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)