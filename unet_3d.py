import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# 3d UNet
# ======================================================
class UNet(nn.Module):
    def __init__(self, in_channels=1, squeeze=False):
        super(UNet, self).__init__()
        n0 = 64 # was 64 initially
        self.conv1 = Conv(in_channels, n0)
        self.down1 = Down(n0, 2*n0)
        self.down2 = Down(2*n0, 4*n0)
        self.down3 = Down(4*n0, 8*n0)
        self.down4 = Down(8*n0, 8*n0)
        self.up1 = Up(8*n0, 4*n0)
        self.up2 = Up(4*n0, 2*n0)
        self.up3 = Up(2*n0, n0)
        self.up4 = Up(n0, n0)
        self.out = OutConv(n0, 1)
        self.squeeze = squeeze

    def forward(self, x):
        if self.squeeze:
            x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        if self.squeeze:
            x = x.squeeze(1)
        return x

# ======================================================
# A convolutional block consisting of two 3d conv layers, each followed by a 3d batch norm and a relu
# ======================================================
class Conv(nn.Module):
    def __init__(self, in_size, out_size):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_size),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(out_size, out_size, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_size),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================================================
# A max-pool3d by a factor of 2, followed by the conv block defined above.
# ======================================================
class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.MaxPool3d(2),
                                  Conv(in_size, out_size))

    def forward(self, x):
        x = self.down(x)
        return x

# ======================================================
# Takes two inputs.
# The first input is passed through a 3d transpose conv, 
# the output of this transpose conv is concatenated with the second input along the channel dimension
# this is now passed through the conv block defined above
# ======================================================
class Up(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=2, stride=2)
        self.conv = Conv(in_size * 2, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        out = torch.cat([up, x2], dim=1)
        out = self.conv(out)
        return out

# ======================================================
# A 3d conv layer, without batch norm or activation function
# ======================================================
class OutConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x
