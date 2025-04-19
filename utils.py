
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ✅ Custom embedding model used during training
class ChartEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.convnet(x).view(x.size(0), -1)
        return self.fc(x)

# ✅ Load the custom model and apply weights
def load_model():
    model = ChartEmbeddingNet()
    path = os.path.join("data", "model_weights.pth")
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

# ✅ Use the model to extract an embedding from an input image
def extract_embedding(img: Image.Image) -> torch.Tensor:
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb
