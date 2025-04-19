import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
WEIGHTS_PATH = os.path.join("data", "model_weights.pth")
EMBEDDINGS_PATH = os.path.join("data", "embeddings.pth")

# AWS/S3 configuration
AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")

# Initialize S3 client
s3_client = None
if AWS_REGION and S3_BUCKET_NAME and S3_WEIGHTS_KEY:
    s3_client = boto3.client("s3", region_name=AWS_REGION)

def download_weights():
    if s3_client:
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        if not os.path.isfile(WEIGHTS_PATH):
            s3_client.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

class ChartEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3,32,5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32)
        )
    def forward(self,x):
        x = self.convnet(x).view(x.size(0),-1)
        return self.fc(x)

_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.isfile(WEIGHTS_PATH):
            download_weights()
        model = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_embedding(img: Image.Image) -> torch.Tensor:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb
