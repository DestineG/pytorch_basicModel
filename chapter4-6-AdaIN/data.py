# chapter-4-6-AdaIN/data.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.content_images = [f for f in os.listdir(content_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.style_images = [f for f in os.listdir(style_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"找到 {len(self.content_images)} 张内容图像")
        print(f"找到 {len(self.style_images)} 张风格图像")
    
    def __len__(self):
        return 2000
    
    def __getitem__(self, idx):
        # 循环使用图像
        content_path = os.path.join(self.content_dir, 
                                   self.content_images[idx % len(self.content_images)])
        style_path = os.path.join(self.style_dir, 
                                 self.style_images[idx % len(self.style_images)])
        
        # 加载图像
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        
        return content_img, style_img


def get_transform(img_size=512):
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])