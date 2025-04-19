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

# AWS/S3 configuration from environment
AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")

# Initialize S3 client
_s3_client = None
if AWS_REGION and S3_BUCKET_NAME and S3_WEIGHTS_KEY:
    _s3_client = boto3.client("s3", region_name=AWS_REGION)

def download_weights_from_s3():
    if _s3_client:
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        if not os.path.isfile(WEIGHTS_PATH):
            _s3_client.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

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
        if not os.path.isfile(WEIGHTS_PATH):
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

def extract_embedding(img: Image.Image) -> torch.Tensor:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb

def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    w, h = query_img.size
    mid = w // 2
    # initial overlay with red line
    left = query_img.crop((0, 0, mid, h))
    match_resized = match_img.resize((w, h))
    overlay = Image.new("RGB", (w, h))
    overlay.paste(left, (0,0))
    overlay.paste(match_resized.crop((mid,0,w,h)), (mid,0))
    draw0 = ImageDraw.Draw(overlay)
    draw0.line([(mid,0),(mid,h)], fill=(255,0,0), width=2)
    # blue line logic
    arr = np.array(overlay)
    red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    x_split = red_cols[-1] if len(red_cols) >= 2 else int(np.argmax(prop_red))
    left_region  = overlay.crop((0, 0, x_split+1, h))
    right_region = overlay.crop((x_split+1, 0, w, h))
    rh, rw = right_region.size[1], right_region.size[0]
    arr_r = np.array(right_region)
    y0, y1 = int(0.10*rh), int(0.90*rh)
    gray = np.dot(arr_r[y0:y1], [0.299,0.587,0.114])
    dark = gray < 200
    min_pix = int(0.01*(y1-y0))
    coords = [(x, int(np.median(np.where(dark[:,x])[0])) + y0)
              for x in range(rw) if len(np.where(dark[:,x])[0]) >= min_pix]
    arr_l = np.array(left_region)
    gray_l = np.dot(arr_l[y0:y1], [0.299,0.587,0.114])
    dark_l = gray_l < 200
    ys_l = np.where(dark_l[:, -2])[0]
    if len(ys_l) >= min_pix:
        y_med_l = int(np.median(ys_l)) + y0
    else:
        y_med_l = int((y0+y1)/2)
    full_coords = [(0, y_med_l)] + coords
    blank = Image.new("RGB", (rw, rh), (255,255,255))
    draw1 = ImageDraw.Draw(blank)
    if full_coords:
        draw1.line(full_coords, fill=(0,0,255), width=3)
    combined = Image.new("RGB", (w,h))
    combined.paste(left_region, (0,0))
    combined.paste(blank, (x_split+1,0))
    cx, cy = int(0.10*w), int(0.10*h)
    return combined.crop((cx, cy, w-cx, h-cy))
