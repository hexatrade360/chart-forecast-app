import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ─── Device and Paths ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH    = os.path.join("data", "model_weights.pth")
EMBEDDINGS_PATH = os.path.join("data", "embeddings.pth")

# ─── AWS / S3 Configuration ───────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")
_s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else None

def download_weights_from_s3():
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if _s3 and not os.path.isfile(WEIGHTS_PATH):
        _s3.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

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
    global _model
    if _model is None:
        if not os.path.isfile(WEIGHTS_PATH):
            download_weights_from_s3()
        m = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _model = m
    return _model

# ─── Preprocessing ────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img: Image.Image) -> torch.Tensor:
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return load_model()(t).squeeze().cpu()

def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    w, h = query_img.size

    # 1) crop central 80% to focus on candles
    x0, x1 = int(0.1*w), int(0.9*w)
    y0, y1 = int(0.1*h), int(0.9*h)
    body = query_img.crop((x0, y0, x1, y1))
    arr = np.array(body)
    R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # 2) mask pure-red or pure-green candles
    red   = (R > 200) & (G < 100) & (B < 100)
    green = (G > 200) & (R < 100) & (B < 100)
    mask  = red | green
    counts = mask.sum(axis=0)
    minpix = int(0.01 * (y1-y0))
    candle_cols = np.where(counts > minpix)[0]

    # 3) last candle column (relative), map back to full width
    rel = candle_cols[-1] if candle_cols.size else (x1-x0)//2
    x_split = x0 + int(rel)

    # 4) paste forecast region
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w, h)).convert("RGBA")
    fc = m.crop((x_split, 0, w, h))
    mask_fc = fc.split()[-1].point(lambda p: 128)
    overlay.paste(fc, (x_split, 0), mask_fc)

    # 5) draw red split
    d = ImageDraw.Draw(overlay)
    d.line([(x_split,0),(x_split,h)], fill=(255,0,0), width=2)

    # 6) trace blue line (unchanged)
    left  = overlay.crop((0,0,x_split,h))
    right = overlay.crop((x_split,0,w,h))
    ar = np.array(right)
    rh, rw = ar.shape[:2]
    y0r, y1r = int(0.1*rh), int(0.9*rh)
    gray = np.dot(ar[y0r:y1r], [0.299,0.587,0.114])
    dark = gray < 200
    mp = int(0.01*(y1r-y0r))
    coords = [
        (x, int(np.median(np.where(dark[:,x])[0]))+y0r)
        for x in range(rw) 
        if len(np.where(dark[:,x])[0]) >= mp
    ]
    al = np.array(left)
    grayl = np.dot(al[y0r:y1r], [0.299,0.587,0.114])
    darkl = grayl < 200
    ys = np.where(darkl[:, -2])[0]
    ymed = int(np.median(ys))+y0r if ys.size>=mp else (y0r+y1r)//2
    full = [(0,ymed)] + coords
    blank = Image.new("RGB",(rw,rh),(255,255,255))
    d2 = ImageDraw.Draw(blank)
    if full: d2.line(full, fill=(0,0,255), width=3)

    # 7) reassemble & final 10% crop
    combo = Image.new("RGB",(w,h))
    combo.paste(left.convert("RGB"),(0,0))
    combo.paste(blank,       (x_split,0))
    cx, cy = int(0.1*w), int(0.1*h)
    return combo.crop((cx,cy,w-cx,h-cy))
