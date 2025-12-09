# chapter4-5-FCN8S/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # 下采样分支（当尺寸或通道不一致时）
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out

def make_layer(in_channels, out_channels, blocks, stride=1):
    layers = []
    # first block: may downsample
    layers.append(Bottleneck(in_channels, out_channels, stride))
    # next blocks
    for _ in range(1, blocks):
        layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))
    return nn.Sequential(*layers)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # stem：输入 → 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet50 layers: (3, 4, 6, 3)
        self.layer1 = make_layer(64,  64, blocks=3, stride=1)  # 256 out
        self.layer2 = make_layer(64 * Bottleneck.expansion, 128, blocks=4, stride=2) # 512 out
        self.layer3 = make_layer(128 * Bottleneck.expansion, 256, blocks=6, stride=2) # 1024 out
        self.layer4 = make_layer(256 * Bottleneck.expansion, 512, blocks=3, stride=2) # 2048 out

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # H/2
        x = self.maxpool(x)                     # H/4

        x1 = self.layer1(x)  # 256 channels, H/4
        x2 = self.layer2(x1) # 512 channels, H/8
        x3 = self.layer3(x2) # 1024 channels, H/16
        x4 = self.layer4(x3) # 2048 channels, H/32

        return x2, x3, x4  # skip connections needed for FCN-8s

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        self.backbone = ResNet50Backbone()

        # 1×1 conv 得到分类分数
        self.score4 = nn.Conv2d(2048, num_classes, kernel_size=1)   # layer4
        self.score3 = nn.Conv2d(1024, num_classes, kernel_size=1)   # layer3
        self.score2 = nn.Conv2d(512, num_classes, kernel_size=1)    # layer2

        # 上采样部分（权重可学习）
        self.up2  = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.up4  = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.up8  = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)

    def forward(self, x):
        # backbone 输出 1/8, 1/16, 1/32
        x2, x3, x4 = self.backbone(x)

        s4 = self.score4(x4)  # 1/32
        s3 = self.score3(x3)  # 1/16
        s2 = self.score2(x2)  # 1/8

        x = self.up2(s4)      # 1/32 → 1/16
        x = x + s3            # skip

        x = self.up4(x)       # 1/16 → 1/8
        x = x + s2            # skip

        x = self.up8(x)       # 1/8 → 1/1

        return x

if __name__ == "__main__":
    model = FCN8s(num_classes=21)
    inp = torch.randn(1, 3, 224, 224)
    if inp.shape[2] % 32 != 0 or inp.shape[3] % 32 != 0:
        raise ValueError("Input height and width must be multiples of 32.")
    out = model(inp)
    print(out.shape)  # (1, 21, 224, 224)
