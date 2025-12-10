# chapter4.6-CIN/model.py

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
from pathlib import Path


# ---------------------------
# Utils: image dataset + transforms
# ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = list(Path(folder).glob('*'))
        self.transform = transform

    def __len__(self):
        return max(1, len(self.paths))

    def __getitem__(self, idx):
        if len(self.paths) == 0:
            raise RuntimeError("No images in folder")
        p = self.paths[idx % len(self.paths)]
        img = Image.open(p).convert('RGB')
        return self.transform(img)

def load_img(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    to_tensor = transforms.ToTensor()
    return to_tensor(img).unsqueeze(0)

def save_img(tensor, path):
    tensor = tensor.detach().cpu().clamp(0,1)
    utils.save_image(tensor, path)

class CIN(nn.Module):
    def __init__(self, num_features, num_styles):
        super().__init__()
        self.num_styles = num_styles

        # 每一种风格都有独立的 gamma / beta
        self.gamma = nn.Parameter(torch.ones(num_styles, num_features))
        self.beta = nn.Parameter(torch.zeros(num_styles, num_features))

        # 普通 InstanceNorm，不带 learnable params
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, style_id):
        """
        x: BxCxHxW
        style_id: int or tensor, 指定使用哪种风格
        """
        out = self.norm(x)

        # gamma, beta: 1xCx1x1，expand 到 B
        gamma = self.gamma[style_id].view(1, -1, 1, 1)
        beta = self.beta[style_id].view(1, -1, 1, 1)

        return out * gamma + beta

class ResidualBlockCIN(nn.Module):
    def __init__(self, channels, num_styles):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.cin1 = CIN(channels, num_styles)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.cin2 = CIN(channels, num_styles)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style_id):
        residual = x
        out = self.relu(self.cin1(self.conv1(x), style_id))
        out = self.cin2(self.conv2(out), style_id)
        return out + residual

class StyleTransferCIN(nn.Module):
    def __init__(self, num_styles):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 9, padding=4)
        self.cin1 = CIN(32, num_styles)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.cin2 = CIN(64, num_styles)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.cin3 = CIN(128, num_styles)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlockCIN(128, num_styles) for _ in range(5)
        ])

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dconv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.cin4 = CIN(64, num_styles)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.cin5 = CIN(32, num_styles)

        self.conv_out = nn.Conv2d(32, 3, 9, padding=4)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, style_id):
        y = self.relu(self.cin1(self.conv1(x), style_id))
        y = self.relu(self.cin2(self.conv2(y), style_id))
        y = self.relu(self.cin3(self.conv3(y), style_id))

        for res in self.res_blocks:
            y = res(y, style_id)

        y = self.relu(self.cin4(self.dconv1(self.up1(y)), style_id))
        y = self.relu(self.cin5(self.dconv2(self.up2(y)), style_id))
        y = self.conv_out(y)

        return y

def train():
    pass

if __name__ == "__main__":
    x = torch.randn(4, 3, 256, 256)  # batch size 4
    style_id = 2  # 使用风格 2
    model = StyleTransferCIN(num_styles=5)
    out = model(x, style_id)
    print(out.shape)  # 应该是 [4, 3, 256, 256]