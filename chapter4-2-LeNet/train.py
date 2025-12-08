from tqdm import tqdm
import torch

from .data import get_mnist_dataloader
from .model import LeNet


def get_trainComposers(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    return optimizer, criterion

def validate(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    val_loss = 0.0
    p = tqdm(val_loader, desc="Validation", ascii=True)
    with torch.no_grad():
        for images, labels in p:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
    return avg_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, device):
    model.to(device)
    num_epochs = 1000

    val_best_loss, val_best_accuracy = float('inf'), 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 每个 epoch 一个 tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", ascii=True)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新 tqdm 后缀
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}", flush=True)

        # 每 N epoch 进行验证
        if (epoch + 1) % 10 == 0:
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%", flush=True)
            if val_accuracy > val_best_accuracy:
                val_best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'chapter4-2-LeNet/checkpoints/best_model.pth')
                print(f"New best model saved with accuracy: {val_best_accuracy:.2f}%", flush=True)

if __name__ == "__main__":
    train_loader = get_mnist_dataloader(batch_size=64, train=True)
    val_loader = get_mnist_dataloader(batch_size=64, train=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LeNet()
    optimizer, criterion = get_trainComposers(model)
    
    train(model, train_loader, val_loader, optimizer, criterion, device)