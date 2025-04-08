import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize

# Match the CNN structure used in your Colab training script
class SwissCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path="mnist_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwissCNN()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

def predict_digit(model, image):
    device = next(model.parameters()).device
    tensor = ToTensor()(image).unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]
    tensor = Normalize((0.5,), (0.5,))(tensor)

    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()
