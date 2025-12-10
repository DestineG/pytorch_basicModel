# chapter-4-6-AdaIN/train.py

import torch
from torch.utils.data import DataLoader
import os
import argparse

from model import AdaINModel, train
from data import StyleTransferDataset, get_transform


def get_opt():
    parser = argparse.ArgumentParser(description='训练AdaIN风格迁移模型')
    parser.add_argument('--content_dir', type=str, required=True,
                       help='内容图像目录路径')
    parser.add_argument('--style_dir', type=str, required=True,
                       help='风格图像目录路径')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小 (默认: 4)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数 (默认: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率 (默认: 0.0001)')
    parser.add_argument('--content_weight', type=float, default=1.0,
                       help='内容损失权重 (默认: 1.0)')
    parser.add_argument('--style_weight', type=float, default=10.0,
                       help='风格损失权重 (默认: 10.0)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='风格强度 (默认: 1.0)')
    parser.add_argument('--img_size', type=int, default=512,
                       help='图像尺寸 (默认: 512)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数 (默认: 4)')
    parser.add_argument('--save_path', type=str, default='./checkpoints/adain_model',
                       help='模型保存路径 (默认: ./checkpoints/adain_model)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用设备 (默认: cuda)')
    
    args = parser.parse_args()
    return args

def main():
    args = get_opt()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 创建数据集和数据加载器
    transform = get_transform()
    dataset = StyleTransferDataset(
        args.content_dir, 
        args.style_dir, 
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # 创建模型
    model = AdaINModel(encoder_pretrained=True)
    
    # 开始训练
    train(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        alpha=args.alpha,
        device=args.device,
        save_path=args.save_path
    )


# python train.py --content_dir G:\datasets\coco2014\train --style_dir G:\datasets\coco2014\style
if __name__ == '__main__':
    main()
