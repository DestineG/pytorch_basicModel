# chapter4-3-InceptionNet-v3/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.pool_features = pool_features

        self.branch1_conv1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch2_conv1x1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch2_conv5x5 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3_conv1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3_conv3x3_1 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3_conv3x3_2 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1x1 = BasicConv2d(in_channels, pool_features, kernel_size=1)
    
    def get_outchannels(self):
        return 64 + 64 + 96 + self.pool_features
    
    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv5x5(branch2)

        branch3 = self.branch3_conv1x1(x)
        branch3 = self.branch3_conv3x3_1(branch3)
        branch3 = self.branch3_conv3x3_2(branch3)

        branch4 = self.branch4_pool(x)
        branch4 = self.branch4_conv1x1(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.in_channels = in_channels

        self.branch1_conv3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, padding=1)

        self.branch2_conv1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch2_conv3x3_1 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch2_conv3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)

        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def get_outchannels(self):
        return 384 + 96 + self.in_channels

    def forward(self, x):
        branch1 = self.branch1_conv3x3(x)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv3x3_1(branch2)
        branch2 = self.branch2_conv3x3_2(branch2)

        branch3 = self.branch3_pool(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()

        self.branch1_conv1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch2_conv1x1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch2_conv7x7_1 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_conv7x7_2 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch3_conv1x1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch3_conv7x1_1 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_conv1x7_1 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_conv7x1_2 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_conv1x7_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
    
    def get_outchannels(self):
        return 192 + 192 + 192 + 192
    
    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv7x7_1(branch2)
        branch2 = self.branch2_conv7x7_2(branch2)

        branch3 = self.branch3_conv1x1(x)
        branch3 = self.branch3_conv7x1_1(branch3)
        branch3 = self.branch3_conv1x7_1(branch3)
        branch3 = self.branch3_conv7x1_2(branch3)
        branch3 = self.branch3_conv1x7_2(branch3)

        branch4 = self.branch4_pool(x)
        branch4 = self.branch4_conv1x1(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.in_channels = in_channels

        self.branch1_conv1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch1_conv3x3 = BasicConv2d(192, 320, kernel_size=3, stride=2, padding=1)

        self.branch2_conv1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch2_conv1x7 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_conv7x1 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch2_conv3x3 = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)

        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def get_outchannels(self):
        return 320 + 192 + self.in_channels
    
    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)
        branch1 = self.branch1_conv3x3(branch1)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv1x7(branch2)
        branch2 = self.branch2_conv7x1(branch2)
        branch2 = self.branch2_conv3x3(branch2)

        branch3 = self.branch3_pool(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.in_channels = in_channels

        self.branch1_conv1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch2_conv1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch2_conv3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch2_conv1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))

        self.branch3_conv1x1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3_conv3x1 = BasicConv2d(448, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3_conv1x3 = BasicConv2d(448, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_conv3x3 = BasicConv2d(384 + 384, 384, kernel_size=3, padding=1)

        self.branch4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
    
    def get_outchannels(self):
        return 320 + (384 + 384) + 384 + 192
    
    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)

        branch2 = self.branch2_conv1x1(x)
        branch2a = self.branch2_conv3x1(branch2)
        branch2b = self.branch2_conv1x3(branch2)
        branch2 = torch.cat([branch2a, branch2b], 1)

        branch3 = self.branch3_conv1x1(x)
        branch3a = self.branch3_conv3x1(branch3)
        branch3b = self.branch3_conv1x3(branch3)
        branch3 = torch.cat([branch3a, branch3b], 1)
        branch3 = self.branch3_conv3x3(branch3)

        branch4 = self.branch4_pool(x)
        branch4 = self.branch4_conv1x1(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = nn.Parameter(torch.tensor(0.01))
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = nn.Parameter(torch.tensor(0.001))
    
    def forward(self, x):
        x = nn.functional.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 192, 28, 28)
    A = InceptionA(in_channels=192, pool_features=32)
    output = A(x)
    print("A: Input shape:", x.shape, " Output shape:", output.shape, A.get_outchannels())
    B = InceptionB(in_channels=192)
    output = B(x)
    print("B: Input shape:", x.shape, " Output shape:", output.shape, B.get_outchannels())
    C = InceptionC(in_channels=192, channels_7x7=128)
    output = C(x)
    print("C: Input shape:", x.shape, " Output shape:", output.shape, C.get_outchannels())
    D = InceptionD(in_channels=192)
    output = D(x)
    print("D: Input shape:", x.shape, " Output shape:", output.shape, D.get_outchannels())
    E = InceptionE(in_channels=192)
    output = E(x)
    print("E: Input shape:", x.shape, " Output shape:", output.shape, E.get_outchannels())
    aux = InceptionAux(in_channels=192, num_classes=10)
    output = aux(x)
    print("Aux: Input shape:", x.shape, " Output shape:", output.shape)