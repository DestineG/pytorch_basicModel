# chapter-4-6-AdaIN/inference.py

import torch
from PIL import Image
from torchvision import transforms
from model import AdaINModel


def single_inference(content_path, style_path, model_path=None, img_size=512, alpha=1.0, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载内容图和风格图
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    content_img = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).to(device)  # [1,3,H,W]
    style_img = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(device)

    # 创建模型
    model = AdaINModel(encoder_pretrained=True).to(device)
    model.eval()

    # 加载训练好的权重（如果有）
    if model_path is not None and torch.cuda.is_available():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_path}")

    with torch.no_grad():
        output = model(content_img, style_img, alpha=alpha)
    
    # 输出转换成PIL图像
    output_img = output.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_img.clamp(0, 1))
    
    return output_img


if __name__ == "__main__":
    content_name = "avril"
    # weight_name = "adain_model_final"
    weight_name = "adain_model_epoch_90"
    content_path = f"./figures/content/{content_name}.jpg"
    style_name_list = ["1", "2", "3", "4", "5", "6"]
    for style_name in style_name_list:
        style_path = f"./figures/style/{style_name}.jpg"
        model_path = f"checkpoints/{weight_name}.pth"
        output_img = single_inference(content_path, style_path, model_path, img_size=512, alpha=1.0)
        output_img.save(f"./figures/output/{weight_name}_{content_name}_to_{style_name}.jpg")  # 保存