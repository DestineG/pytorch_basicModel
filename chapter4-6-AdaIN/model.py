# chapter-4-6-AdaIN/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


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


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN)
    将内容特征图的统计信息（均值和方差）调整为风格特征图的统计信息
    """
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat):
        """
        Args:
            content_feat: 内容特征图 [B, C, H, W]
            style_feat: 风格特征图 [B, C, H, W]
        Returns:
            adain_feat: 自适应归一化后的特征图 [B, C, H, W]
        """
        assert content_feat.size()[:2] == style_feat.size()[:2], \
            "Content and style features must have the same number of channels"
        
        # 计算均值和方差
        content_mean, content_std = self._calc_mean_std(content_feat)
        style_mean, style_std = self._calc_mean_std(style_feat)
        
        # 归一化内容特征
        normalized_content = (content_feat - content_mean) / (content_std + self.eps)
        
        # 应用风格统计信息
        adain_feat = normalized_content * style_std + style_mean
        
        return adain_feat

    def _calc_mean_std(self, feat, eps=1e-5):
        """
        计算特征图的均值和标准差
        Args:
            feat: 特征图 [B, C, H, W]
        Returns:
            mean: 均值 [B, C, 1, 1]
            std: 标准差 [B, C, 1, 1]
        """
        size = feat.size()
        assert len(size) == 4
        B, C = size[:2]
        
        # 计算每个通道的均值和方差
        feat_reshaped = feat.view(B, C, -1)  # [B, C, H*W]
        mean = feat_reshaped.mean(dim=2, keepdim=True)  # [B, C, 1]
        std = feat_reshaped.std(dim=2, keepdim=True) + eps  # [B, C, 1]
        
        # 扩展维度以匹配原始特征图
        mean = mean.view(B, C, 1, 1)
        std = std.view(B, C, 1, 1)
        
        return mean, std


class Decoder(nn.Module):
    """
    解码器：将特征图转换回图像
    使用反卷积和上采样层来恢复图像尺寸
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # 解码器结构：512 -> 256 -> 128 -> 64 -> 3
        self.decoder = nn.Sequential(
            # 第一层：512 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # 第二层：256 -> 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # 第三层：128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # 第四层：64 -> 3 (输出RGB图像)
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, padding=0),
            nn.Sigmoid()  # 将输出限制在[0, 1]范围
        )

    def forward(self, x):
        return self.decoder(x)


class AdaINModel(nn.Module):
    """
    完整的AdaIN风格迁移模型
    包含编码器（VGG19）和解码器
    """
    def __init__(self, encoder_pretrained=True):
        super(AdaINModel, self).__init__()
        self.encoder = VGG19(input_channels=3, pretrained=encoder_pretrained)
        self.adain = AdaIN()
        self.decoder = Decoder()
        
        # 冻结编码器参数，只训练解码器
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, x):
        """使用编码器提取特征"""
        return self.encoder(x)

    def decode(self, x):
        """使用解码器将特征转换为图像"""
        return self.decoder(x)

    def forward(self, content, style, alpha=1.0):
        """
        前向传播
        Args:
            content: 内容图像 [B, 3, H, W]
            style: 风格图像 [B, 3, H, W]
            alpha: 风格强度，0-1之间，默认1.0
        Returns:
            output: 风格迁移后的图像 [B, 3, H, W]
        """
        # 提取特征
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # AdaIN归一化
        adain_feat = self.adain(content_feat, style_feat)
        
        # 混合内容和风格特征（通过alpha控制风格强度）
        if alpha < 1.0:
            t = alpha * adain_feat + (1 - alpha) * content_feat
        else:
            t = adain_feat
        
        # 解码生成图像
        output = self.decode(t)
        
        return output


def calc_content_loss(input_feat, target_feat):
    """
    计算内容损失（L2损失）
    Args:
        input_feat: 输入特征图
        target_feat: 目标特征图
    Returns:
        loss: 内容损失
    """
    return F.mse_loss(input_feat, target_feat)


def calc_style_loss(input_feat, target_feat):
    """
    计算风格损失（基于均值和方差的L2损失）
    Args:
        input_feat: 输入特征图
        target_feat: 目标特征图
    Returns:
        loss: 风格损失
    """
    input_mean, input_std = calc_mean_std(input_feat)
    target_mean, target_std = calc_mean_std(target_feat)
    
    mean_loss = F.mse_loss(input_mean, target_mean)
    std_loss = F.mse_loss(input_std, target_std)
    
    return mean_loss + std_loss


def calc_mean_std(feat, eps=1e-5):
    """
    计算特征图的均值和标准差
    Args:
        feat: 特征图 [B, C, H, W]
    Returns:
        mean: 均值 [B, C, 1, 1]
        std: 标准差 [B, C, 1, 1]
    """
    size = feat.size()
    assert len(size) == 4
    B, C = size[:2]
    
    feat_reshaped = feat.view(B, C, -1)
    mean = feat_reshaped.mean(dim=2, keepdim=True)
    std = feat_reshaped.std(dim=2, keepdim=True) + eps
    
    mean = mean.view(B, C, 1, 1)
    std = std.view(B, C, 1, 1)
    
    return mean, std


def train_step(model, content_images, style_images, optimizer, 
               content_weight=1.0, style_weight=10.0, alpha=1.0, device='cuda'):
    """
    训练一步
    Args:
        model: AdaIN模型
        content_images: 内容图像批次 [B, 3, H, W]
        style_images: 风格图像批次 [B, 3, H, W]
        optimizer: 优化器
        content_weight: 内容损失权重
        style_weight: 风格损失权重
        alpha: 风格强度
        device: 设备
    Returns:
        total_loss: 总损失
        content_loss_val: 内容损失值
        style_loss_val: 风格损失值
    """
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    output = model(content_images, style_images, alpha=alpha)
    
    # 提取特征用于计算损失
    output_feat = model.encode(output)
    content_feat = model.encode(content_images)
    style_feat = model.encode(style_images)
    
    # 计算内容损失
    content_loss = calc_content_loss(output_feat, content_feat) * content_weight
    
    # 计算风格损失
    style_loss = calc_style_loss(output_feat, style_feat) * style_weight
    
    # 总损失
    total_loss = content_loss + style_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), content_loss.item(), style_loss.item()


def train(model, train_loader, num_epochs, lr=0.0001, content_weight=1.0, 
          style_weight=10.0, alpha=1.0, device='cuda', save_path=None):
    """
    训练函数
    Args:
        model: AdaIN模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        content_weight: 内容损失权重
        style_weight: 风格损失权重
        alpha: 风格强度
        device: 设备
        save_path: 模型保存路径
    """
    model = model.to(device)
    
    # 只优化解码器参数
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    
    print(f"开始训练，共 {num_epochs} 轮")
    print(f"设备: {device}")
    print(f"学习率: {lr}")
    print(f"内容损失权重: {content_weight}, 风格损失权重: {style_weight}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        num_batches = 0
        
        for batch_idx, (content_images, style_images) in enumerate(train_loader):
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            
            # 训练一步
            total_loss, content_loss, style_loss = train_step(
                model, content_images, style_images, optimizer,
                content_weight, style_weight, alpha, device
            )
            
            epoch_total_loss += total_loss
            epoch_content_loss += content_loss
            epoch_style_loss += style_loss
            num_batches += 1
            
            # 每10个batch打印一次
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {total_loss:.4f} (Content: {content_loss:.4f}, Style: {style_loss:.4f})")
        
        # 计算平均损失
        avg_total_loss = epoch_total_loss / num_batches
        avg_content_loss = epoch_content_loss / num_batches
        avg_style_loss = epoch_style_loss / num_batches
        
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成 - "
              f"平均损失: {avg_total_loss:.4f} "
              f"(内容: {avg_content_loss:.4f}, 风格: {avg_style_loss:.4f})")
        print("-" * 50)
        
        # 保存模型
        if save_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, f"{save_path}_epoch_{epoch+1}.pth")
            print(f"模型已保存到 {save_path}_epoch_{epoch+1}.pth")
    
    print("训练完成！")
    
    # 保存最终模型
    if save_path:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total_loss,
        }, f"{save_path}_final.pth")
        print(f"最终模型已保存到 {save_path}_final.pth")

def test():
    # 测试代码
    print("测试AdaIN模型组件...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试AdaIN层
    print("\n1. 测试AdaIN层...")
    adain = AdaIN()
    content_feat = torch.randn(2, 512, 32, 32)
    style_feat = torch.randn(2, 512, 32, 32)
    adain_output = adain(content_feat, style_feat)
    print(f"AdaIN输出形状: {adain_output.shape}")
    
    # 测试解码器
    print("\n2. 测试解码器...")
    decoder = Decoder()
    decoder_output = decoder(content_feat)
    print(f"解码器输出形状: {decoder_output.shape}")
    
    # 测试完整模型
    print("\n3. 测试完整AdaIN模型...")
    model = AdaINModel(encoder_pretrained=True)
    model = model.to(device)
    
    content_img = torch.randn(2, 3, 256, 256).to(device)
    style_img = torch.randn(2, 3, 256, 256).to(device)
    
    output = model(content_img, style_img, alpha=1.0)
    print(f"模型输出形状: {output.shape}")
    
    # 测试损失函数
    print("\n4. 测试损失函数...")
    output_feat = model.encode(output)
    content_feat = model.encode(content_img)
    style_feat = model.encode(style_img)
    
    content_loss = calc_content_loss(output_feat, content_feat)
    style_loss = calc_style_loss(output_feat, style_feat)
    print(f"内容损失: {content_loss.item():.4f}")
    print(f"风格损失: {style_loss.item():.4f}")
    
    print("\n所有测试通过！")


if __name__ == "__main__":
    test()