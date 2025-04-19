
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

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

def load_model():
    model = ChartEmbeddingNet()
    weights_path = "data/model_weights.pth"
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def extract_embedding(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)
    emb = load_model()(tensor).squeeze().cpu()
    return emb

def extract_split_point(img):
    arr = np.array(img)
    h, w, _ = arr.shape
    red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    return red_cols[-1] if len(red_cols) >= 2 else int(np.argmax(prop_red))
