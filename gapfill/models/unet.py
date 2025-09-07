import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(inplace=True),
    )

class UNetSmall(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.inc = conv_block(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv_block(64,128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv_block(128,256))
        self.bot = conv_block(256,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = conv_block(256,128)
        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec1 = conv_block(128,64)
        self.outc = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xb = self.bot(x3)
        x = self.up2(xb); x = torch.cat([x, x2], dim=1); x = self.dec2(x)
        x = self.up1(x);  x = torch.cat([x, x1], dim=1); x = self.dec1(x)
        return self.outc(x)
