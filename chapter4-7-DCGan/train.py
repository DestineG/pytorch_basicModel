# chapter4-7-DCGan/train.py

import os
from tdqm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import DCGANGenerator, DCGANDiscriminator, initialize_weights


def get_data_loader(data_root: str, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def train(
    epochs: int = 500,
    batch_size: int = 128,
    latent_dim: int = 100,
    lr: float = 2e-4,
    beta1: float = 0.5,
    data_root: str = "./data",
    out_dir: str = "./outputs",
    device: str | torch.device = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dataloader = get_data_loader(data_root, batch_size)

    netG = DCGANGenerator(latent_dim=latent_dim, img_channels=1, feature_maps=64).to(device)
    netD = DCGANDiscriminator(img_channels=1, feature_maps=64).to(device)
    netG.apply(initialize_weights)
    netD.apply(initialize_weights)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    for epoch in range(epochs):
        p = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False, ascii=True)
        for i, (imgs, _) in enumerate(p):
            real_imgs = imgs.to(device)
            b_size = real_imgs.size(0)

            # 判别器优化
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            netD.zero_grad()
            output_fake = netD(fake_imgs.detach())
            output_real = netD(real_imgs)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            lossD_fake = criterion(output_fake, fake_labels)
            lossD_real = criterion(output_real, real_labels)
            total_lossD = lossD_real + lossD_fake
            total_lossD.backward()
            optimizerD.step()

            # 生成器优化
            netG.zero_grad()
            output_fake_for_g = netD(fake_imgs)
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            lossG = criterion(output_fake_for_g, real_labels)  # real_labels = 1
            lossG.backward()
            optimizerG.step()

            if i % 100 == 0:
                p.set_postfix({
                    "Loss_D": (lossD_real + lossD_fake).item(),
                    "Loss_G": lossG.item(),
                })

        # 每个 epoch 结束保存样本和模型
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(fake, os.path.join(out_dir, f"fake_samples_epoch_{epoch+1:03d}.png"), normalize=True)
        torch.save(netG.state_dict(), os.path.join(out_dir, f"netG_epoch_{epoch+1:03d}.pth"))
        torch.save(netD.state_dict(), os.path.join(out_dir, f"netD_epoch_{epoch+1:03d}.pth"))


if __name__ == "__main__":
    train()

