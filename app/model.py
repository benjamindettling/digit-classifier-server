import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize

# Match the CNN structure used in your Colab training script
class SwissCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
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
    tensor = ToTensor()(image).unsqueeze(0).to(device) 
    tensor = Normalize((0.9492,), (0.1168,))(tensor)

    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()
