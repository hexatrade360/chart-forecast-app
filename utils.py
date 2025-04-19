
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class ChartEmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.convnet(x).view(x.size(0), -1)
        return self.fc(x)

def load_model(weights_path='data/model_weights.pth'):
    model = ChartEmbeddingNet()
    state = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

def extract_embedding(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    model = load_model()
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb
