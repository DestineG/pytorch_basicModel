from PIL import Image

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mnist_dataloader(batch_size=64, train=True, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=train, transform=transform, download=download)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    return dataloader

def get_data(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image

if __name__ == "__main__":
    input = get_data("./figures/18.jpg")
    print(input.shape)