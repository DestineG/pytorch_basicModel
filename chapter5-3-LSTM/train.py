# chapter5-3-LSTM/train.py

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import LSTM


def generate_sine_wave_dataset(seq_length=10, num_samples=1000, noise_level=0.0):
    """
    生成正弦波数据集
    
    Args:
        seq_length: 输入序列长度
        num_samples: 样本数量
        noise_level: 噪声水平
    
    Returns:
        X: 输入序列 (num_samples, seq_length, 1)
        y: 目标值 (num_samples, 1)
    """
    X = []
    y = []
    
    for i in range(num_samples):
        # 生成不同频率和相位的正弦波
        t_start = np.random.uniform(0, 2 * np.pi)
        frequency = np.random.uniform(0.5, 2.0)
        
        # 生成序列
        t = np.linspace(t_start, t_start + seq_length * 0.1, seq_length)
        sequence = np.sin(frequency * t)
        
        # 添加噪声
        if noise_level > 0:
            sequence += np.random.normal(0, noise_level, sequence.shape)
        
        # 目标值是序列的下一个值
        t_next = t_start + seq_length * 0.1
        target = np.sin(frequency * t_next)
        
        X.append(sequence)
        y.append(target)
    
    X = np.array(X).reshape(num_samples, seq_length, 1)
    y = np.array(y).reshape(num_samples, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)


def train_model(model, train_loader, criterion, optimizer, num_epochs=100, device='cpu'):
    """训练模型"""
    model.train()
    losses = []
    p = tqdm(range(num_epochs), desc="训练进度", ascii=True)
    for epoch in p:
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 3 == 0:
            p.set_postfix({'Loss': avg_loss})
    
    return losses


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    return mse, mae, predictions, targets


def visualize_results(predictions, targets, losses, num_samples=100):
    """Visualize results"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot predictions vs targets
    axes[0].plot(targets[:num_samples], label='Target', alpha=0.7)
    axes[0].plot(predictions[:num_samples], label='Prediction', alpha=0.7)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Prediction vs Target')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss curve
    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("Results saved to training_results.png")
    plt.show()


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 超参数
    seq_length = 20
    input_size = 1
    hidden_size = 64
    output_size = 1
    num_layers = 2
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    num_train_samples = 2000
    num_test_samples = 500
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 生成数据集
    print("生成训练数据集...")
    train_X, train_y = generate_sine_wave_dataset(
        seq_length=seq_length, 
        num_samples=num_train_samples,
        noise_level=0.05
    )
    
    print("生成测试数据集...")
    test_X, test_y = generate_sine_wave_dataset(
        seq_length=seq_length, 
        num_samples=num_test_samples,
        noise_level=0.05
    )
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 创建模型
    model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("\n开始训练...")
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # 评估模型
    print("\n评估模型...")
    mse, mae, predictions, targets = evaluate_model(model, test_loader, device)
    print(f'测试集 MSE: {mse:.6f}')
    print(f'测试集 MAE: {mae:.6f}')
    
    # 可视化结果
    visualize_results(predictions, targets, losses)
    
    # 保存模型
    torch.save(model.state_dict(), f'{checkpoint_dir}/lstm_model.pth')
    print(f"\n模型已保存到 {checkpoint_dir}/lstm_model.pth")


if __name__ == "__main__":
    main()
