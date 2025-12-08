# chapter4-3-ResNet/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# BasicBlock（ResNet18/34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样分支（当尺寸或通道不一致时）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out


# Bottleneck（ResNet50/101/152）
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


# ResNet18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # (B, 3, H, W) -> (B, 64, H/2, W/2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # (B, 64, H/2, W/2) -> (B, 64, H/2, W/2)
        self.bn1 = nn.BatchNorm2d(64)
        # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # (B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # (B, 256, H/16, W/16) -> (B, 512, H/32, W/32)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # (B, 512, H/32, W/32) -> (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # (B, 512) -> (B, num_classes)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # 第一个 block 可能需要 downsample
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 后续 block stride = 1
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ResNet50
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # (B, 3, H, W) -> (B, 64, H/2, W/2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # (B, 64, H/2, W/2) -> (B, 64, H/2, W/2)
        self.bn1 = nn.BatchNorm2d(64)
        # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (B, 64, H/4, W/4) -> (B, 256, H/4, W/4)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        # (B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
        self.layer2 = self._make_layer(64 * Bottleneck.expansion, 128, 4, stride=2)
        # (B, 512, H/8, W/8) -> (B, 1024, H/16, W/16)
        self.layer3 = self._make_layer(128 * Bottleneck.expansion, 256, 6, stride=2)
        # (B, 1024, H/16, W/16) -> (B, 2048, H/32, W/32)
        self.layer4 = self._make_layer(256 * Bottleneck.expansion, 512, 3, stride=2)
        # (B, 2048, H/32, W/32) -> (B, 2048, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # (B, 2048) -> (B, num_classes)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # 第一个 block 可能需要 downsample
        layers.append(Bottleneck(in_channels, out_channels, stride))
        # 后续 block stride = 1
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# --------------------------------------------------
# 测试一下模型是否可运行
# --------------------------------------------------
if __name__ == "__main__":
    res18 = ResNet18(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    out = res18(x)
    print(out.shape)  # torch.Size([1, 10])

    res50 = ResNet50(num_classes=10)
    out = res50(x)
    print(out.shape)  # torch.Size([1, 10])