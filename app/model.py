import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path="mnist_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MNIST_CNN()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

def predict_digit(model, image):
    device = next(model.parameters()).device
    tensor = ToTensor()(image)
    tensor = Normalize((0.1307,), (0.3081,))(tensor)
    tensor = tensor.unsqueeze(0).to(device)  # shape [1, 1, 28, 28]
    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()
