import os
import boto3
import pickle
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AWS/S3 configuration
AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")
WEIGHTS_PATH   = os.path.join("data", "model_weights.pth")

# Initialize S3 client
_s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else None

def download_weights_from_s3():
    if _s3 and not os.path.isfile(WEIGHTS_PATH):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        print(f"📥 Downloading weights from s3://{S3_BUCKET_NAME}/{S3_WEIGHTS_KEY}")
        _s3.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)
        print(f"✅ Model weights saved to {WEIGHTS_PATH}")

def crop_chart_body(img):
    w, h = img.size
    left, right = int(0.1*w), int(0.9*w)
    top, bottom = int(0.1*h), int(0.9*h)
    return img.crop((left, top, right, bottom))

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

_model = None

def load_model():
    global _model
    if _model is None:
        download_weights_from_s3()
        model = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img):
    img_body = crop_chart_body(img)
    tensor = transform(img_body).unsqueeze(0).to(device)
    model = load_model()
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb
