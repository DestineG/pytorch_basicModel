# chapter4-7-AVE/model.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# VAE做的是将输入压缩到一个稠密的向量簇，这个向量簇由于稠密的性质就有可能具有线性插值的能力
# 稠密是由于VAE的KL散度（正则化损失）强制实现的
# 若收敛时abs(mu) >> abs(std)，则说明潜空间构造比较失败，反之则比较成功
# 比较KL和正则化的构造能力(samples_kl_ld10 VS samples_reg_ld10)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值的全连接层
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差的全连接层
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # 返回均值和对数方差
    
    # 以可微分方式随机采样潜在变量
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # 采样潜在变量
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 重构输入
    
    def forward(self, x):
        # shape: [batch_size, 1, 28, 28] -> [batch_size, 784] -> [batch_size, latent_dim]
        mu, logvar = self.encode(x.view(-1, 784))
        # shape: [batch_size, latent_dim] -> [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)
        # shape: [batch_size, latent_dim] -> [batch_size, 784]
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # 使用重构误差 + KL 散度作为 VAE 损失
    bce = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction="sum"
    )
    # 将(mu, std)约束到(0, 1) std=exp(logvar/2) 收敛到0.02，-3.68
    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return bce + kld, bce, kld

    # 同样将(mu, logvar)约束到(0, 1) 收敛到-0.12，-1.32
    reg = 0.5 * torch.sum(mu.pow(2) + logvar.pow(2))
    return bce + reg, bce, reg


def train_vae(epochs=5, batch_size=128, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
    )

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ascii=True)
        for data, _ in progress:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d} | train_loss: {avg_loss:.4f}")

        # 阶段性可视化：仅保存图片示例
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                data, _ = next(iter(val_loader))
                data = data.to(device)

                # 这里取出 mu 和 logvar
                recon_batch, mu, logvar = model(data)

                # 计算 batch 的均值
                mu_mean = float(mu.mean().cpu())
                logvar_mean = float(logvar.mean().cpu())

                mu_str = f"{mu_mean:.3f}"
                logvar_str = f"{logvar_mean:.3f}"

                os.makedirs("samples", exist_ok=True)

                # 保存输入图像
                save_image(
                    data[:64].cpu(),
                    f"samples/epoch_{epoch:02d}_mu_{mu_str}_logvar_{logvar_str}_input.png",
                    nrow=8
                )

                # 保存重建图像
                save_image(
                    recon_batch[:64].view(-1, 1, 28, 28).cpu(),
                    f"samples/epoch_{epoch:02d}_mu_{mu_str}_logvar_{logvar_str}_recon.png",
                    nrow=8,
                )


if __name__ == "__main__":
    # 运行一个简单的训练流程示例
    train_vae(epochs=300, batch_size=256)