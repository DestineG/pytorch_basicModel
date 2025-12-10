import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19(nn.Module):
    """
    只保留到 relu4_1 的 VGG19 编码器，用于提取内容/风格特征
    """
    def __init__(self, input_channels=3, pretrained=True):
        super().__init__()
        # 与 AdaIN 版本保持一致的轻量配置
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
        self.features = nn.Sequential(*layers)

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        c_idx = p_idx = 0
        while c_idx < len(self.features) and p_idx < len(pretrained):
            c_layer = self.features[c_idx]
            p_layer = pretrained[p_idx]

            if isinstance(c_layer, nn.Conv2d) and isinstance(p_layer, nn.Conv2d):
                c_layer.weight.data.copy_(p_layer.weight.data)
                if p_layer.bias is not None:
                    c_layer.bias.data.copy_(p_layer.bias.data)
                c_idx += 1
                p_idx += 1
                continue

            # 跳过非卷积层的对应关系
            if isinstance(c_layer, (nn.ReLU, nn.MaxPool2d)):
                c_idx += 1
            else:
                p_idx += 1

    def forward(self, x):
        return self.features(x)


def whiten_and_color(content_feat, style_feat, eps=1e-5):
    """
    Whitening and Coloring Transform (WCT)
    将内容特征通过白化+上色对齐到风格特征的统计分布
    Args:
        content_feat: [B, C, H, W]
        style_feat:   [B, C, H, W]
    Returns:
        transformed feature: [B, C, H, W]
    """
    assert content_feat.shape[:2] == style_feat.shape[:2], "内容/风格特征通道数需一致"

    B, C, H, W = content_feat.shape
    c_feat = content_feat.view(B, C, -1)
    s_feat = style_feat.view(B, C, -1)

    out = torch.zeros_like(c_feat)

    eye = torch.eye(C, device=content_feat.device)

    for b in range(B):
        c = c_feat[b]
        s = s_feat[b]

        c_mean = c.mean(dim=1, keepdim=True)
        s_mean = s.mean(dim=1, keepdim=True)

        c_centered = c - c_mean
        s_centered = s - s_mean

        # 协方差
        c_cov = (c_centered @ c_centered.t()) / (c_centered.size(1) - 1) + eps * eye
        s_cov = (s_centered @ s_centered.t()) / (s_centered.size(1) - 1) + eps * eye

        # 对称矩阵特征分解
        c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov)
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov)

        # 白化：E * D^{-1/2} * E^T * (x - mu_c)
        c_d_inv_sqrt = torch.diag(torch.rsqrt(torch.clamp(c_eigvals, min=eps)))
        whiten = c_eigvecs @ c_d_inv_sqrt @ c_eigvecs.t() @ c_centered

        # 上色：E_s * D_s^{1/2} * E_s^T * whiten + mu_s
        s_d_sqrt = torch.diag(torch.sqrt(torch.clamp(s_eigvals, min=eps)))
        colored = s_eigvecs @ s_d_sqrt @ s_eigvecs.t() @ whiten

        out[b] = colored + s_mean

    return out.view(B, C, H, W)


class Decoder(nn.Module):
    """
    反卷积解码器，与 VGG19 编码器的大致对称结构
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, padding=0),
            nn.Sigmoid()  # 输出归一化到 [0,1]
        )

    def forward(self, x):
        return self.decoder(x)


class WCTModel(nn.Module):
    """
    基于单层（relu4_1）的 WCT 风格迁移模型
    - 使用预训练 VGG19 编码器提取特征
    - 对内容特征做 WCT 与风格特征对齐
    - 通过解码器生成风格化图像
    """
    def __init__(self, encoder_pretrained=True):
        super().__init__()
        self.encoder = VGG19(input_channels=3, pretrained=encoder_pretrained)
        self.decoder = Decoder()

        # 冻结编码器
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, content, style, alpha=1.0):
        """
        Args:
            content: [B,3,H,W] 内容图像
            style:   [B,3,H,W] 风格图像
            alpha:   风格强度，1 表示完全使用 WCT，0 表示保持原内容
        """
        c_feat = self.encode(content)
        s_feat = self.encode(style)

        t = whiten_and_color(c_feat, s_feat)

        if alpha < 1.0:
            t = alpha * t + (1 - alpha) * c_feat

        return self.decode(t)


def stylize(model, content_img, style_img, alpha=1.0, device=None):
    """
    便利函数：输入原图与风格图，返回风格化结果
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        content_img = content_img.to(device)
        style_img = style_img.to(device)
        output = model(content_img, style_img, alpha=alpha)
    return output


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"测试 WCT，使用设备: {device}")

    model = WCTModel(encoder_pretrained=True).to(device)
    content = torch.randn(2, 3, 256, 256, device=device)
    style = torch.randn(2, 3, 256, 256, device=device)

    with torch.no_grad():
        out = model(content, style, alpha=0.8)

    print(f"输入: {content.shape}, 输出: {out.shape}, 值域 [{out.min():.3f}, {out.max():.3f}]")


if __name__ == "__main__":
    test()
