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

_s3 = None
if AWS_REGION and S3_BUCKET_NAME and S3_WEIGHTS_KEY:
    _s3 = boto3.client("s3", region_name=AWS_REGION)

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

def extract_embedding(img: Image.Image) -> torch.Tensor:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb

def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    # image dims
    w, h = query_img.size

    # 1) define chart-body crop (10% margins)
    left_cut, right_cut = int(0.1*w), int(0.9*w)
    top_cut, bottom_cut = int(0.1*h), int(0.9*h)
    body = query_img.crop((left_cut, top_cut, right_cut, bottom_cut))
    arr = np.array(body)

    # 2) detect ANY non‑white pixel as candle (R<240 or G<240 or B<240)
    mask = np.any(arr < 240, axis=2)
    col_counts = mask.sum(axis=0)
    minpix = int(0.01 * (bottom_cut - top_cut))
    valid = col_counts  # already just body-width columns
    candle_cols = np.where(valid > minpix)[0]

    # 3) relative split in body, then absolute in full image
    if len(candle_cols):
        rel_split = candle_cols[-1]
    else:
        rel_split = (right_cut - left_cut)//2
    x_split = left_cut + int(rel_split)

    # 4) build overlay: paste forecast region
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w, h)).convert("RGBA")
    fc = m.crop((x_split, 0, w, h))
    mask_fc = fc.split()[-1].point(lambda p: 128)
    overlay.paste(fc, (x_split, 0), mask_fc)

    # 5) draw red line
    draw = ImageDraw.Draw(overlay)
    draw.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # 6) trace blue forecast line
    left  = overlay.crop((0, 0, x_split, h))
    right = overlay.crop((x_split, 0, w, h))
    arr_r = np.array(right)
    rh, rw = arr_r.shape[:2]
    y0, y1 = int(0.1*rh), int(0.9*rh)
    gray = np.dot(arr_r[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    minp = int(0.01*(y1-y0))

    coords = [
        (x, int(np.median(np.where(dark[:,x])[0]))+y0)
        for x in range(rw)
        if len(np.where(dark[:,x])[0]) >= minp
    ]

    arr_l = np.array(left)
    gray_l = np.dot(arr_l[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys = np.where(dark_l[:, -2])[0]
    ymed = int(np.median(ys))+y0 if len(ys)>=minp else (y0+y1)//2
    full = [(0, ymed)] + coords

    blank = Image.new("RGB", (rw, rh), (255, 255, 255))
    d2 = ImageDraw.Draw(blank)
    if full:
        d2.line(full, fill=(0,0,255), width=3)

    # 7) reassemble & final crop
    combined = Image.new("RGB", (w, h))
    combined.paste(left.convert("RGB"), (0,0))
    combined.paste(blank,  (x_split,0))
    cx, cy = int(0.1*w), int(0.1*h)
    return combined.crop((cx, cy, w-cx, h-cy))
