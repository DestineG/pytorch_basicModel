import torch
import torch.nn.functional as F

from .data import get_data
from .model import LeNet


def infer(image_path, model_path, device='cpu'):
    # Load and preprocess the image
    input_image = get_data(image_path).to(device)
    
    # Initialize the model and load weights
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        print(output)
        print(F.softmax(output, dim=1))
        predicted_class = output.argmax(dim=1).item()
    
    return predicted_class

if __name__ == "__main__":
    image_path = "chapter4-2-LeNet/figures/6.png"
    model_path = "chapter4-2-LeNet/checkpoints/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    predicted_class = infer(image_path, model_path, device)
    print(f"Predicted class: {predicted_class}")