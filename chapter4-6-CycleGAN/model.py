# chapter4-6-CycleGAN/model.py

import torch
import torch.nn as nn

from data import create_dataloader


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock, self).__init__()
        padding = dilation
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果通道不同，就加 1×1 卷积做 residual 映射
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out + residual

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CycleGANGenerator, self).__init__()
        # 定义CycleGAN生成器的网络结构
        self.block1 = nn.Sequential(
            ResBlock(in_channels, 16, 1),
            ResBlock(16, 32, 1),
            ResBlock(32, 32, 1)
        )
        self.block2 = nn.Sequential(
            ResBlock(in_channels, 16, 2),
            ResBlock(16, 32, 2),
            ResBlock(32, 32, 2)
        )
        self.block3 = nn.Sequential(
            ResBlock(in_channels, 16, 4),
            ResBlock(16, 32, 4),
            ResBlock(32, 32, 4)
        )
        self.block4 = nn.Sequential(
            ResBlock(in_channels, 16, 8),
            ResBlock(16, 32, 8),
            ResBlock(32, 32, 8)
        )
        self.decoder = nn.Sequential(
            ResBlock(128, 64, 1),
            ResBlock(64, 32, 1),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.decoder(out)
        return out

class CycleGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator (CycleGAN 标准结构)
    输入:  [B, 3, H, W]
    输出: [B, 1, H/8, W/8] 的真/假评分
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(CycleGANDiscriminator, self).__init__()

        # 不使用归一化的第一层（论文规范）
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 不下采样，保持 receptive field 为 70×70
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出 patch map
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class CycleGANModel(nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()
        self.G_A2B = CycleGANGenerator()
        self.G_B2A = CycleGANGenerator()
        self.D_A = CycleGANDiscriminator()
        self.D_B = CycleGANDiscriminator()

    def forward(self, x_A, x_B):
        fake_B = self.G_A2B(x_A)
        fake_A = self.G_B2A(x_B)
        return fake_B, fake_A

def train_step(model, real_A, real_B, optim_D, optim_G, D_criterion, cyc_criterion):
    fake_B, fake_A = model(real_A, real_B)

    # --------- 判别器优化 ---------
    optim_D.zero_grad()
    # 判别器输出
    D_A_real = model.D_A(real_A)
    D_A_fake = model.D_A(fake_A.detach())
    D_B_real = model.D_B(real_B)
    D_B_fake = model.D_B(fake_B.detach())
    D_A_real_label = torch.ones_like(D_A_real)
    D_A_fake_label = torch.zeros_like(D_A_fake)
    D_B_real_label = torch.ones_like(D_B_real)
    D_B_fake_label = torch.zeros_like(D_B_fake)
    # 判别器损失
    loss_D_A = (D_criterion(D_A_real, D_A_real_label) + D_criterion(D_A_fake, D_A_fake_label)) * 0.5
    loss_D_B = (D_criterion(D_B_real, D_B_real_label) + D_criterion(D_B_fake, D_B_fake_label)) * 0.5
    total_D_loss = loss_D_A + loss_D_B
    total_D_loss.backward()
    optim_D.step()

    # --------- 生成器优化 ---------
    optim_G.zero_grad()
    # 对抗损失
    D_A_fake = model.D_A(fake_A)
    D_B_fake = model.D_B(fake_B)
    D_B_fake_label = torch.ones_like(D_B_fake)
    D_A_fake_label = torch.ones_like(D_A_fake)
    loss_G_A2B = D_criterion(D_B_fake, D_B_fake_label)
    loss_G_B2A = D_criterion(D_A_fake, D_A_fake_label)
    loss_G = loss_G_A2B + loss_G_B2A
    # 循环一致性损失
    cyc_B, cyc_A = model(fake_A, fake_B)
    loss_cycle_A = cyc_criterion(cyc_A, real_A)
    loss_cycle_B = cyc_criterion(cyc_B, real_B)
    loss_cycle = loss_cycle_A + loss_cycle_B
    total_G_loss = loss_G + loss_cycle
    total_G_loss.backward()
    optim_G.step()

    return total_D_loss.item(), total_G_loss.item()

def train(model, dataloader, optim_D, optim_G):
    D_criterion = nn.MSELoss()
    cyc_criterion = nn.L1Loss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100
    model.to(device)
    for epoch in range(num_epochs):
        for iter, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            real_A = batch['real_A']
            real_B = batch['real_B']
            loss_D, loss_G = train_step(model, real_A, real_B, optim_D, optim_G, D_criterion, cyc_criterion)
            if (iter + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]  Iter [{iter+1}/{len(dataloader)}]  Loss_D: {loss_D:.4f}  Loss_G: {loss_G:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/cyclegan_epoch_{epoch+1}.pth')
    
    torch.save(model.state_dict(), 'checkpoints/cyclegan_final.pth')

def test():
    # 测试CycleGAN生成器和判别器的前向传播
    G = CycleGANGenerator()
    D = CycleGANDiscriminator()

    x = torch.randn(2, 3, 256, 256)  # 假设输入为2张256x256的RGB图像

    fake_images = G(x)
    print("Generator output shape:", fake_images.shape)  # 应该是 [2, 3, 256, 256]

    D_out = D(fake_images)
    print("Discriminator output shape:", D_out.shape)  # 应该是 [2, 1, 30, 30] (256/8=32, 减去边界效应)

if __name__ == "__main__":
    dir_A = "./data/domain_A"
    dir_B = "./data/domain_B"
    dataloader = create_dataloader(
        dir_A=dir_A,
        dir_B=dir_B,
        img_size=512,
        batch_size=4,
        load_to = 'memory'
    )
    model = CycleGANModel()
    optim_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optim_G = torch.optim.Adam(
        list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    train(model, dataloader, optim_D, optim_G)