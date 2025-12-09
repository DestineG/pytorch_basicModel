# chapter4-5-UNet/model.py

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.double_conv(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((skip_connection, x), dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = UNetEncoder(64, 128)
        self.enc3 = UNetEncoder(128, 256)
        self.enc4 = UNetEncoder(256, 512)

        self.bottleneck = UNetEncoder(512, 1024)

        self.dec4 = UNetDecoder(1024, 512)
        self.dec3 = UNetDecoder(512, 256)
        self.dec2 = UNetDecoder(256, 128)
        self.dec1 = UNetDecoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        skip4 = self.enc4(skip3)

        bottleneck = self.bottleneck(skip4)

        dec4 = self.dec4(bottleneck, skip4)
        dec3 = self.dec3(dec4, skip3)
        dec2 = self.dec2(dec3, skip2)
        dec1 = self.dec1(dec2, skip1)

        out = self.final_conv(dec1)
        return out

if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    inp = torch.randn(1, 3, 224, 224)
    if inp.shape[2] % 16 != 0 or inp.shape[3] % 16 != 0:
        raise ValueError("Input height and width must be multiples of 16.")
    out = model(inp)
    print(out.shape)  # (1, 1, 224, 224)