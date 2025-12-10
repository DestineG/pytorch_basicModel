# chapter4-6-Ulyanov/model.py

import os
import argparse
from PIL import Image
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, utils
from torchvision.models import VGG19_Weights
from torch.utils.data import DataLoader, Dataset


# ---------------------------
# Utils: image dataset + transforms
# ---------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = list(Path(folder).glob('*'))
        self.transform = transform

    def __len__(self):
        return max(1, len(self.paths))

    def __getitem__(self, idx):
        if len(self.paths) == 0:
            raise RuntimeError("No images in folder")
        p = self.paths[idx % len(self.paths)]
        img = Image.open(p).convert('RGB')
        return self.transform(img)

def load_img(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    to_tensor = transforms.ToTensor()
    return to_tensor(img).unsqueeze(0)

def save_img(tensor, path):
    tensor = tensor.detach().cpu().clamp(0,1)
    utils.save_image(tensor, path)

# ---------------------------
# Gram matrix (style)
# ---------------------------
def gram_matrix(y):
    # y: BxCxHxW
    (b, c, h, w) = y.size()
    features = y.view(b, c, h*w)
    G = torch.bmm(features, features.transpose(1,2))  # B x C x C
    return G / (c * h * w)

# ---------------------------
# Generator: encoder - residuals - decoder
# ---------------------------
def conv_layer(in_c, out_c, kernel=3, stride=1, padding=1, relu=True, inst_norm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding)]
    if inst_norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            conv_layer(channels, channels, kernel=3, stride=1, padding=1, relu=True),
            conv_layer(channels, channels, kernel=3, stride=1, padding=1, relu=False)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_res=5):
        super().__init__()
        # Encoder
        self.enc1 = conv_layer(in_channels, ngf, kernel=9, stride=1, padding=4)
        self.enc2 = conv_layer(ngf, ngf*2, kernel=3, stride=2, padding=1)
        self.enc3 = conv_layer(ngf*2, ngf*4, kernel=3, stride=2, padding=1)

        # Transformer (residuals)
        res_blocks = []
        for _ in range(n_res):
            res_blocks.append(ResidualBlock(ngf*4))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Decoder / Upsample
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_layer(ngf*4, ngf*2, kernel=3, stride=1, padding=1)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_layer(ngf*2, ngf, kernel=3, stride=1, padding=1)
        )
        self.out = nn.Sequential(
            nn.Conv2d(ngf, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res_blocks(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.out(x)
        # Tanh -> rescale to [0,1]
        return (x + 1) / 2

# ---------------------------
# VGG descriptor for losses
# ---------------------------
class VGGDescriptor(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        # We will slice vgg and expose outputs at indices specified
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x, target_layers):
        # return dict of layer_idx -> features (after ReLU)
        features = {}
        v = x
        for i, layer in enumerate(self.vgg):
            v = layer(v)
            if i in target_layers:
                features[i] = v
        return features

# ---------------------------
# Training / Loss functions
# ---------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])

    # Dataset
    dataset = ImageFolderDataset(args.content_dir, transform) if args.content_dir else None
    if dataset:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    else:
        dataloader = None

    # Style image
    style_img = load_img(args.style_image, size=(args.image_size, args.image_size)).to(device)

    # Model
    G = Generator(ngf=args.ngf, n_res=args.n_res).to(device)
    vgg_descriptor = VGGDescriptor(content_layers=args.content_layers, style_layers=args.style_layers).to(device)
    for p in vgg_descriptor.parameters():
        p.requires_grad = False

    # Prepare targets by forwarding style image and optionally a content example
    style_targets = vgg_descriptor(style_img, args.style_layers)
    style_grams = {k: gram_matrix(v) for k, v in style_targets.items()}

    # optimizer
    optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)

    # optional resume
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("Loading checkpoint", args.checkpoint)
        ck = torch.load(args.checkpoint, map_location=device)
        G.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck.get('opt', optimizer.state_dict()))
        start_epoch = ck.get('epoch', 0)

    iters = 0
    print("Start training...")
    for epoch in range(start_epoch, args.epochs):
        if dataloader is None:
            # Single image mode: feed the same (or content_image if specified)
            content_example = load_img(args.content_image, size=(args.image_size, args.image_size)).to(device)
            loader_iter = [(content_example,)]
        else:
            loader_iter = dataloader
        pbar = tqdm(loader_iter, desc=f"Epoch {epoch+1}/{args.epochs}", ascii=True)
        for batch in pbar:
            if dataloader is not None:
                content_batch = batch.to(device)
            else:
                content_batch = batch[0].to(device)

            optimizer.zero_grad()

            # forward generator
            output = G(content_batch)

            # content features for content loss (use conv4_2 -> index mapping example)
            # user provided target_layers as list of indices; pick content layer only
            content_feats_gen = vgg_descriptor(output, args.content_layers)
            content_feats_orig = vgg_descriptor(content_batch, args.content_layers)
            # style features for style loss (use style_layers)
            style_feats_gen = vgg_descriptor(output, args.style_layers)

            # content loss (MSE on selected content layer)
            Lc = 0.0
            for k in args.content_layers:
                Lc = Lc + F.mse_loss(content_feats_gen[k], content_feats_orig[k])

            # style loss (MSE on Gram matrices)
            Ls = 0.0
            for k in args.style_layers:
                Gg = gram_matrix(style_feats_gen[k])
                Gs = style_grams[k].expand_as(Gg) if Gg.size(0) != style_grams[k].size(0) else style_grams[k]
                Ls = Ls + F.mse_loss(Gg, Gs)

            loss = args.content_weight * Lc + args.style_weight * Ls
            loss.backward()
            optimizer.step()

            iters += 1
            if iters % args.log_interval == 0:
                pbar.set_postfix_str(f"Loss {loss.item():.4f} (Lc {Lc.item():.4f} Ls {Ls.item():.4f})")

            if iters % args.save_interval == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, f"sample_{iters}.jpg")
                save_img(output[0], out_path)
                print("Saved sample", out_path)
        pbar.close()

        # end epoch: save checkpoint
        ck = {'model': G.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch+1}
        torch.save(ck, args.save_model or os.path.join(args.output_dir, f"ckpt_epoch{epoch+1}.pth"))
        print("Saved checkpoint epoch", epoch+1)

    print("Training finished. Saving model...")
    torch.save({'model': G.state_dict()}, args.save_model or os.path.join(args.output_dir, "model_final.pth"))
    print("Saved model.")

# ---------------------------
# Eval / stylize single image with checkpoint
# ---------------------------
def stylize_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(ngf=args.ngf, n_res=args.n_res).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ck['model'] if 'model' in ck else ck)
    G.eval()
    with torch.no_grad():
        content = load_img(args.content_image, size=(args.image_size, args.image_size)).to(device)
        out = G(content)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_img(out[0], args.output)
        print("Saved stylized image to", args.output)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--content_dir', type=str, default=None, help='folder of content images (for training)')
    p.add_argument('--content_image', type=str, default=None, help='single content image (for eval or single-image training)')
    p.add_argument('--style_image', type=str, required=True, help='style image path')
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--save_model', type=str, default='generator.pth')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--eval', action='store_true')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ngf', type=int, default=32)
    p.add_argument('--n_res', type=int, default=5)
    p.add_argument('--content_weight', type=float, default=1.0)
    p.add_argument('--style_weight', type=float, default=1e5)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_interval', type=int, default=100)
    return p.parse_args()


# python -m  model --content_dir /dataroot/liujiang/data/datasets/coco2014/val --style_image /dataroot/liujiang/data/datasets/coco2014/style.jpg --epochs 20 --save_model model.pth --output_dir outputs > train.log
# python -m  model --content_dir /dataroot/liujiang/data/datasets/coco2014/val --style_image /dataroot/liujiang/data/datasets/coco2014/style.jpg --epochs 20 --output_dir outputs > train.log

# python ulyanov_pytorch.py --eval --checkpoint model.pth --content_image ./some.jpg --output outputs/result.jpg
if __name__ == "__main__":
    args = parse_args()
    # default layer indices for vgg19.features as used in Gatys-style:
    # conv4_2 relu index = 22 (content), conv1_1=1 conv2_1=6 conv3_1=11 conv4_1=20 (style)
    args.content_layers = [22]      # use conv4_2 relu (after activation) -> index for features
    args.style_layers = [1, 6, 11, 20]
    if args.eval:
        assert args.checkpoint, "Provide --checkpoint for eval"
        assert args.content_image and args.output, "Provide content and output"
        stylize_single(args)
    else:
        if args.content_dir is None and args.content_image is None:
            raise RuntimeError("Provide content_dir (training) or content_image (single-image train)")
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
