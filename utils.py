import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AWS/S3 configuration
AWS_REGION       = os.environ.get("AWS_REGION")
S3_BUCKET_NAME   = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY   = os.environ.get("S3_WEIGHTS_KEY")
WEIGHTS_PATH     = os.path.join("data", "model_weights.pth")

# Initialize S3 client
_s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION and S3_BUCKET_NAME else None

def download_weights_from_s3():
    """Download model weights from S3 if not already present locally."""
    if _s3:
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        if not os.path.isfile(WEIGHTS_PATH):
            _s3.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

def crop_chart_body(img):
    """Crop 10% margins from each side to isolate the chart area."""
    w, h = img.size
    left, right = int(0.1 * w), int(0.9 * w)
    top, bottom = int(0.1 * h), int(0.9 * h)
    return img.crop((left, top, right, bottom))

class ChartEmbeddingNet(nn.Module):
    """Custom CNN used for training via triplet loss."""
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

_model_cache = None
def load_model():
    """Load and cache the fine-tuned ChartEmbeddingNet weights."""
    global _model_cache
    if _model_cache is None:
        download_weights_from_s3()
        model = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model_cache = model
    return _model_cache

# Preprocessing pipeline used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img):
    """
    Crop chart body, transform, and compute embedding from ChartEmbeddingNet.
    """
    img_body = crop_chart_body(img)
    tensor = transform(img_body).unsqueeze(0).to(device)
    model = load_model()
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb

def generate_overlay_forecast(query_img, match_img):
    """
    Create forecast overlay:
      - Detect red split line in match_img
      - Split query_img left and match_img right
      - Trace forecast line in blue with smoothing
      - Draw red split
      - Crop 10% margins
    """
    w, h = query_img.size
    # Resize match to query dims
    match_r = match_img.resize((w, h))
    arr = np.array(match_r)

    # Detect red vertical split line
    red_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    if len(red_cols) >= 2:
        x_split = int(red_cols[-1])
    else:
        x_split = w // 2
    x_split = max(1, min(x_split, w-1))

    # Split regions
    left = query_img.crop((0, 0, x_split+1, h))
    right = match_r.crop((x_split+1, 0, w, h))

    # Trace forecast line on right
    rh, rw = right.size[1], right.size[0]
    arr_r = np.array(right)
    y0, y1 = int(0.10*rh), int(0.90*rh)
    gray = np.dot(arr_r[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01*(y1-y0))
    coords = []
    for x in range(rw):
        ys = np.where(dark[:, x])[0]
        if len(ys) >= min_pix:
            coords.append((x, int(np.median(ys)) + y0))

    # Continuity on left
    arr_l = np.array(left)
    gray_l = np.dot(arr_l[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys_l = np.where(dark_l[:, -1])[0]
    if len(ys_l) >= min_pix:
        y0l = int(np.median(ys_l)) + y0
    else:
        y0l = (y0 + y1) // 2

    full_coords = [(0, y0l)] + coords

    # Smooth y-coordinates
    xs, ys = zip(*full_coords)
    ys_sm = np.convolve(ys, np.ones(5)/5, mode="same").astype(int)
    full_coords = list(zip(xs, ys_sm))

    # Draw blue forecast line
    blank = Image.new("RGB", (rw, rh), (255, 255, 255))
    draw = ImageDraw.Draw(blank)
    if full_coords:
        draw.line(full_coords, fill=(0, 0, 255), width=3)

    # Combine and draw red split
    combined = Image.new("RGB", (w, h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(blank, (x_split+1, 0))
    draw2 = ImageDraw.Draw(combined)
    draw2.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # Crop margins
    cx, cy = int(0.1*w), int(0.1*h)
    final = combined.crop((cx, cy, w-cx, h-cy))
    return final
