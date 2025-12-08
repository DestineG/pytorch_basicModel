# chapter4-3-ResNet18/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# BasicBlock（ResNet18/34 用的基础残差块）
# --------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道扩展倍数，18/34 为1

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
        identity = x  # 旁路

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out


# --------------------------------------------------
# ResNet18 主体
# --------------------------------------------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 的 layer 配置： [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
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


# --------------------------------------------------
# 测试一下模型是否可运行
# --------------------------------------------------
if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)  # torch.Size([1, 10])
