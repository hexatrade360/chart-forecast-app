
import torch
import torchvision.transforms as transforms
from PIL import Image

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
        return self.fc(self.convnet(x).view(x.size(0), -1))

def load_model():
    model = ChartEmbeddingNet()
    model.load_state_dict(torch.load("data/model_weights.pth", map_location="cpu"))
    model.eval()
    return model

def extract_embedding(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    model = load_model()
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze()
    return emb
