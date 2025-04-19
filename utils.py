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
    """Download model weights from S3 if not present locally."""
    if _s3_client:
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        if not os.path.isfile(WEIGHTS_PATH):
            _s3_client.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

# ─── Model Definition ─────────────────────────────────────────────────────────
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
    """Load and cache the fine-tuned ChartEmbeddingNet weights."""
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

# ─── Preprocessing ────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def crop_chart_body(img: Image.Image) -> Image.Image:
    """No cropping on processing; return full image."""
    return img

def extract_embedding(img: Image.Image) -> torch.Tensor:
    """Compute embedding for a PIL Image."""
    body = crop_chart_body(img)
    tensor = transform(body).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb

# ─── Forecast Overlay ─────────────────────────────────────────────────────────
def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    w, h = query_img.size
    mid = w // 2

    # Split into left (query) and right (forecast)
    left = query_img.crop((0, 0, mid, h))
    match_resized = match_img.resize((w, h))
    right = match_resized.crop((mid, 0, w, h))

    # Trace forecast line on right half
    rh, rw = right.size[1], right.size[0]
    arr = np.array(right)
    y0, y1 = int(0.10*rh), int(0.90*rh)
    gray = np.dot(arr[y0:y1], [0.299, 0.587, 0.114])
    mask = gray < 200
    minpix = int(0.01*(y1-y0))
    coords = [(x, int(np.median(np.where(mask[:,x])[0])) + y0)
              for x in range(rw) if len(np.where(mask[:,x])[0]) >= minpix]

    # Continuity from left half
    arrl = np.array(left)
    grayl = np.dot(arrl[y0:y1], [0.299, 0.587, 0.114])
    maskl = grayl < 200
    ys = np.where(maskl[:, -1])[0]
    if len(ys) >= minpix:
        y0l = int(np.median(ys)) + y0
    else:
        y0l = (y0 + y1) // 2
    full_coords = [(0, y0l)] + coords

    # Smooth line (5-point moving average)
    if full_coords:
        xs, ys = zip(*full_coords)
        ys_sm = np.convolve(ys, np.ones(5)/5, mode='same').astype(int)
        full_coords = list(zip(xs, ys_sm))

    # Draw blue forecast line on blank canvas
    blank = Image.new("RGB", (rw, rh), (255,255,255))
    draw = ImageDraw.Draw(blank)
    if full_coords:
        draw.line(full_coords, fill=(0,0,255), width=3)

    # Combine and draw red split line
    combined = Image.new("RGB", (w, h), (255,255,255))
    combined.paste(left, (0,0))
    combined.paste(blank, (mid,0))
    draw2 = ImageDraw.Draw(combined)
    draw2.line([(mid,0),(mid,h)], fill=(255,0,0), width=2)

    # Final cropping of 10% margins to remove whitespace
    cx, cy = int(0.1*w), int(0.1*h)
    return combined.crop((cx, cy, w-cx, h-cy))
