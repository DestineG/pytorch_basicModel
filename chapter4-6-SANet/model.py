# chapter-4-6-SANet/model.py

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

from data import create_dataloader


class VGG19(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(VGG19, self).__init__()
        self.features = self._make_layers(input_channels)
        if pretrained:
            self.load_weights()
    
    def _make_layers(self, input_channels):
        # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]
        layers = []
        in_channels = input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=False)]
                in_channels = x
        return nn.Sequential(*layers)

    def load_weights(self):
        vgg19_pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        custom_layers = self.features
        idx_p, idx_c = 0, 0
        while idx_p < len(vgg19_pretrained) and idx_c < len(custom_layers):
            layer_p = vgg19_pretrained[idx_p]
            layer_c = custom_layers[idx_c]

            # 匹配卷积层
            if isinstance(layer_p, nn.Conv2d) and isinstance(layer_c, nn.Conv2d):
                layer_c.weight.data.copy_(layer_p.weight.data)
                if layer_p.bias is not None:
                    layer_c.bias.data.copy_(layer_p.bias.data)
                idx_p += 1
                idx_c += 1
            # 跳过自定义网络激活层和池化层
            elif isinstance(layer_c, (nn.ReLU, nn.MaxPool2d)):
                idx_c += 1
            # 跳过预训练网络激活层和池化层
            elif isinstance(layer_p, (nn.ReLU, nn.MaxPool2d)):
                idx_p += 1
            # 跳过其他不匹配层
            else:
                idx_p += 1
                idx_c += 1

    def forward(self, x):
        x = self.features(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # 512 → 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            # Upsample
            nn.Upsample(scale_factor=2, mode='nearest'),

            # 256 → 256 → 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            # Upsample
            nn.Upsample(scale_factor=2, mode='nearest'),

            # 128 → 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            # Upsample
            nn.Upsample(scale_factor=2, mode='nearest'),

            # 64 → 3（输出 RGB）
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class SANet(nn.Module):
    def __init__(self, channels):
        super(SANet, self).__init__()

        # 1×1 conv to get f, g, h
        self.f = nn.Conv2d(channels, channels//2, 1)
        self.g = nn.Conv2d(channels, channels//2, 1)
        self.h = nn.Conv2d(channels, channels, 1)

        # W_out
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, Fc, Fs):
        """
        Fc: content feature  [B, C, H, W]
        Fs: style feature    [B, C, H, W]
        """

        Ff = self.f(Fc)    # [B, C/2, H, W]
        Fg = self.g(Fs)    # [B, C/2, H, W]
        Fh = self.h(Fs)    # [B, C, H, W]

        # reshape: B x C x HW
        B, _, H, W = Ff.shape
        HW = H * W

        Ff = Ff.view(B, -1, HW).permute(0, 2, 1)         # B x HW x C/2
        Fg = Fg.view(B, -1, HW)                          # B x C/2 x HW
        Fh = Fh.view(B, -1, HW)                           # B x C   x HW

        # Attention = softmax(Ff @ Fg / sqrt(d_k))
        # Temperature scaling for better numerical stability
        d_k = Ff.size(-1)
        Attention = torch.bmm(Ff, Fg) / (d_k ** 0.5)     # B x HW x HW
        Attention = F.softmax(Attention, dim=-1)

        # O = Fh @ Attention^T
        Out = torch.bmm(Fh, Attention.permute(0, 2, 1))  # B x C x HW
        Out = Out.view(B, -1, H, W)

        # W_out(Out) + Fc
        Out = self.out_conv(Out) + Fc

        return Out

class SANetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGG19(pretrained=True)
        self.sanet = SANet(512)
        self.decoder = Decoder()

    def forward(self, content, style):
        Fc = self.encoder(content)
        Fs = self.encoder(style)

        Fout = self.sanet(Fc, Fs)
        out = self.decoder(Fout)
        return out

class SANetLoss(nn.Module):
    """SANet 损失函数模块，整合了内容损失、风格损失和总变分损失"""
    def __init__(self, 
                 content_weight=1.0, 
                 style_weight=10.0, 
                 tv_weight=1e-4,
                 content_layers=[18], 
                 style_layers=[4, 9, 18, 27, 36]):
        """
        Args:
            content_weight: 内容损失权重
            style_weight: 风格损失权重
            tv_weight: 总变分损失权重
            content_layers: 用于计算内容损失的层索引
            style_layers: 用于计算风格损失的层索引
        """
        super(SANetLoss, self).__init__()

        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # 用于计算损失的 VGG 特征提取器（需要多层特征）
        vgg_full = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.vgg_features = vgg_full.eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        
        # MSE 损失用于内容损失和风格损失
        self.mse_loss = nn.MSELoss()
    
    def _gram_matrix(self, feat):
        """计算 Gram 矩阵用于风格损失"""
        B, C, H, W = feat.size()
        feat = feat.view(B, C, H * W)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)

    def _get_vgg_features(self, vgg_model, x, layers=None):
        """
        从 VGG 模型中提取多层特征
        layers: 要提取特征的层索引列表，例如 [4, 9, 18, 27, 36] 对应 relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        """
        if layers is None:
            layers = [4, 9, 18, 27, 36]  # 默认提取这些层的特征
        
        features = []
        for i, layer in enumerate(vgg_model):
            x = layer(x)
            if i in layers:
                features.append(x)
        return features
    
    def _content_loss(self, pred_feat, target_feat):
        """内容损失：保持生成图像与内容图像在特征空间中的相似性"""
        return self.mse_loss(pred_feat, target_feat)
    
    def _style_loss(self, pred_feat, target_feat):
        """风格损失：使用 Gram 矩阵匹配风格特征"""
        pred_gram = self._gram_matrix(pred_feat)
        target_gram = self._gram_matrix(target_feat)
        return self.mse_loss(pred_gram, target_gram)
    
    def _tv_loss(self, x):
        """总变分损失：平滑图像，减少噪声"""
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x - 1) * w_x * x.size()[1]
        count_w = h_x * (w_x - 1) * x.size()[1]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def forward(self, output, content, style):
        """
        计算总损失
        Args:
            output: 模型生成的图像 [B, 3, H, W]，范围 [0, 1]
            content: 内容图像 [B, 3, H, W]，范围 [0, 1]
            style: 风格图像 [B, 3, H, W]，范围 [0, 1]
        Returns:
            dict: 包含各项损失的字典
        """
        content_norm = self.transform(content)
        style_norm = self.transform(style)
        output_norm = self.transform(output)
        
        # 提取特征
        content_feats = self._get_vgg_features(self.vgg_features, content_norm, self.content_layers)
        style_feats = self._get_vgg_features(self.vgg_features, style_norm, self.style_layers)
        output_content_feats = self._get_vgg_features(self.vgg_features, output_norm, self.content_layers)
        output_style_feats = self._get_vgg_features(self.vgg_features, output_norm, self.style_layers)
        
        # 内容损失
        content_loss = 0
        for i in range(len(self.content_layers)):
            content_loss += self._content_loss(output_content_feats[i], content_feats[i])
        
        # 风格损失
        style_loss = 0
        for i in range(len(self.style_layers)):
            style_loss += self._style_loss(output_style_feats[i], style_feats[i])
        
        # 总变分损失
        tv_loss = self._tv_loss(output)
        
        # 总损失
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return {
            'total_loss': total_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
            'tv_loss': tv_loss
        }

def train(model, dataloader, optim, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 1000
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = "./log/images"
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    criterion.to(device)
    global_step = 0
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False, ascii=True)
        for iter, batch in enumerate(loop):
            batch = {k: v.to(device) for k, v in batch.items()}
            content = batch['content']
            style = batch['style']

            optim.zero_grad()
            output = model(content, style)
            losses = criterion(output, content, style)
            loss = losses['total_loss']
            loss.backward()
            optim.step()


            # 更新 tqdm 显示的描述信息
            loop.set_postfix({
                'Loss_Total': f'{losses["total_loss"]:.4f}',
                'Loss_Content': f'{losses["content_loss"]:.4f}',
                'Loss_Style': f'{losses["style_loss"]:.4f}',
                'Loss_TV': f'{losses["tv_loss"]:.4f}'
            })

            # 每500 iter保存一次
            if (global_step + 1) % 500 == 0:
                save_image(torch.clamp(content[0], 0, 1), os.path.join(save_dir, f'epoch{epoch+1}_iter{global_step+1}_content.png'))
                save_image(torch.clamp(style[0], 0, 1), os.path.join(save_dir, f'epoch{epoch+1}_iter{global_step+1}_style.png'))
                save_image(torch.clamp(output[0], 0, 1), os.path.join(save_dir, f'epoch{epoch+1}_iter{global_step+1}_output.png'))
            global_step += 1
        loop.close()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'sanet_epoch_{epoch+1}.pth'))
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'sanet_final.pth'))

def test():
    # 测试模型
    content = torch.randn(1, 3, 224, 224)
    style = torch.randn(1, 3, 224, 224)
    model = SANetModel()
    output = model(content, style)
    print(f"Output shape: {output.shape}")
    
    # 测试损失函数
    criterion = SANetLoss(content_weight=1.0, style_weight=10.0, tv_weight=1e-4)
    losses = criterion(output, content, style)
    print(f"\nLosses:")
    print(f"  Total Loss: {losses['total_loss'].item():.4f}")
    print(f"  Content Loss: {losses['content_loss'].item():.4f}")
    print(f"  Style Loss: {losses['style_loss'].item():.4f}")
    print(f"  TV Loss: {losses['tv_loss'].item():.4f}")


if __name__ == "__main__":
    dir_A = "/dataroot/liujiang/data/datasets/DF2K/DF2K_train_HR"
    dir_B = "/dataroot/liujiang/data/datasets/CVCInfrared/train/HR"
    dataloader = create_dataloader(
        dir_A=dir_A,
        dir_B=dir_B,
        img_size=256,
        batch_size=8,
        load_to = 'memory'
    )

    model = SANetModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SANetLoss(content_weight=1.0, style_weight=3e4, tv_weight=1e-4)

    train(model, dataloader, optim, criterion)