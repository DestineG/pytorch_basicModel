# chapter4-6-GatysNST/model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights


class VGG19(nn.Module):
    def __init__(self, input_channels=3):
        super(VGG19, self).__init__()
        self.features = self._make_layers(input_channels)
    
    def _make_layers(self, input_channels):
        # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]
        layers = []
        in_channels = input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=False)]
                in_channels = x
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        return x

# 加载torchvision预训练权重
def load_pretrained_weights(custom_model, verbose=False):

    pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

    custom_layers = custom_model.features
    pre_layers = pretrained

    c_idx = 0
    p_idx = 0

    while True:
        if c_idx >= len(custom_layers) or p_idx >= len(pre_layers):
            break

        c_layer = custom_layers[c_idx]
        p_layer = pre_layers[p_idx]

        if isinstance(c_layer, nn.Conv2d) and isinstance(p_layer, nn.Conv2d):
            if verbose:
                print(f"Copying weights: custom[{c_idx}] <- pretrained[{p_idx}]")

            c_layer.weight.data = p_layer.weight.data.clone()
            c_layer.bias.data   = p_layer.bias.data.clone()

            c_idx += 1
            p_idx += 1
            continue
        
        if isinstance(c_layer, nn.ReLU):
            c_idx += 1
            continue

        if isinstance(c_layer, nn.AvgPool2d):
            c_idx += 1
            continue

        p_idx += 1

    if verbose:
        print("权重加载完成！")

def gram_matrix(feature):
    N, C, H, W = feature.size()
    F = feature.view(C, H*W)
    G = torch.mm(F, F.t())   # C × C
    return G / (C * H * W)   # normalized Gram

loader = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img


def save_image(tensor, path):
    img = tensor.clone().detach().cpu().squeeze(0)
    img = transforms.ToPILImage()(img)
    img.save(path)

def content_loss(gen_feature, content_feature):
    return torch.mean((gen_feature - content_feature)**2)

def style_loss(gen_feature, style_feature):
    Gg = gram_matrix(gen_feature)
    Gs = gram_matrix(style_feature)
    return torch.mean((Gg - Gs)**2)

def extract_features(model, x, target_layers):
    features = {}
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i in target_layers:
            features[i] = x
    return features

# 将待生成图像作为参数使用反向传播进行优化
def run_style_transfer(content_path, style_path, steps=500,
                       alpha=1, beta=100000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load images
    content_img = load_image(content_path).to(device)
    style_img = load_image(style_path).to(device)

    # 2) init generated image
    generated = content_img.clone().requires_grad_(True)

    # 3) load model
    vgg = VGG19()
    load_pretrained_weights(vgg)
    vgg.to(device).eval()

    # ----------------------------------------------
    # 正确的 Gatys 风格迁移特征层
    content_layer = 22
    style_layers = [1, 6, 11, 20]
    target_layers = [content_layer] + style_layers
    # ----------------------------------------------

    # 4) 提取源图像的content features和目标图像的style features
    content_features = extract_features(vgg, content_img, [content_layer])
    style_features   = extract_features(vgg, style_img, style_layers)
    content_features = {k: v.detach() for k, v in content_features.items()}
    style_features   = {k: v.detach() for k, v in style_features.items()}

    optimizer = optim.LBFGS([generated])

    print("开始优化...")

    run = [0]
    while run[0] < steps:

        def closure():
            optimizer.zero_grad()

            gen_features = extract_features(vgg, generated, target_layers)

            # 对齐content features(保持内容不变)
            Lc = content_loss(
                gen_features[content_layer],
                content_features[content_layer] # 来自于源图像
            )

            # 对齐style features(风格迁移)
            Ls = 0
            for k in style_layers:
                Ls += style_loss(
                    gen_features[k],
                    style_features[k] # 来自于目标图像
                )

            loss = alpha * Lc + beta * Ls
            loss.backward()

            if run[0] % 50 == 0:
                print(f"Step {run[0]}  Loss: {loss.item():.4f}")

            run[0] += 1
            return loss

        optimizer.step(closure)

    print("优化完成！")
    return generated


# ---------------------------------------------------
# 8. 运行
# ---------------------------------------------------
if __name__ == "__main__":
    stepsList = [0, 100, 300, 500, 1000, 2000, 3000, 5000, 8000, 10000]
    # stepsList = [0]
    for steps in stepsList:
        out = run_style_transfer(
            content_path="./chapter4-6-GatysNST/figures/content.jpg",
            style_path="./chapter4-6-GatysNST/figures/style.jpg",
            steps=steps
        )
        save_image(out, f"./chapter4-6-GatysNST/figures/result_{steps}.jpg")
        print(f"保存 ./chapter4-6-GatysNST/figures/result_{steps}.jpg")
