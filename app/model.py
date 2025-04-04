import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Lambda

class DeepNeuralNet(nn.Module):
    def __init__(self, input_width, hidden_layer_profile, output_width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_layer_profile[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_profile[0], hidden_layer_profile[1]),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_layer_profile[1], output_width)
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.layers(input)
        return self.output_activation(self.output_layer(x))

def load_model(path="mnist_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepNeuralNet(28*28, [512, 256], 10)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

def predict_digit(model, image):
    device = next(model.parameters()).device
    tensor = ToTensor()(image).flatten().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        return output.argmax(dim=1).item()
