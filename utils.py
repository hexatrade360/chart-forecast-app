# utils.py

import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# ─── Configuration ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH    = os.path.join("data", "model_weights.pth")
EMBEDDINGS_PATH = os.path.join("data", "embeddings.pth")

AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")

# Initialize S3 client if credentials are provided
_s3 = boto3.client("s3", region_name=AWS_REGION) if (AWS_REGION and S3_BUCKET_NAME and S3_WEIGHTS_KEY) else None

# ─── Model Definition ──────────────────────────────────────────────────────────
class ChartEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 128)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)

_model = None
def load_model():
    global _model
    if _model is None:
        # download weights if needed
        if _s3 and not os.path.isfile(WEIGHTS_PATH):
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            _s3.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)
        model = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        # strip any prefix if present
        state = {k.split("backbone.")[-1]:v for k,v in state.items()}
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model

# ─── Preprocessing ─────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_embedding(img: Image.Image) -> torch.Tensor:
    """
    Crop center 80%, resize, and pass through embedding network.
    """
    w,h = img.size
    left, right = int(0.1*w), int(0.9*w)
    top, bottom = int(0.1*h), int(0.9*h)
    body = img.crop((left, top, right, bottom))
    tensor = _transform(body).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb

# ─── Overlay + Forecast Line ───────────────────────────────────────────────────
def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    """
    Paste the matching forecast region onto the query,
    draw a dynamic red split at the end of query's candles,
    then overlay a blue forecast line.
    """
    import numpy as _np

    w, h = query_img.size

    # 1) detect last candle column in query (any pixel <240 in any channel)
    arr = _np.array(query_img.convert("RGB"))
    mask = _np.any(arr < 240, axis=2)
    col_counts = mask.sum(axis=0)
    minpix = int(0.01 * h)
    candle_cols = _np.where(col_counts > minpix)[0]
    x_split = (candle_cols[-1] + 1) if candle_cols.size else (w // 2)

    # 2) prepare overlay canvas
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w, h)).convert("RGBA")

    # 3) paste forecast region
    fc = m.crop((x_split, 0, w, h))
    mask_fc = fc.split()[-1].point(lambda p: 128)
    overlay.paste(fc, (x_split, 0), mask_fc)

    # 4) draw red split line
    draw = ImageDraw.Draw(overlay)
    draw.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # 5) trace blue line on forecast region
    left  = overlay.crop((0, 0, x_split, h))
    right = overlay.crop((x_split, 0, w, h))
    ar = _np.array(right)
    rh, rw = ar.shape[:2]
    y0, y1 = int(0.1*rh), int(0.9*rh)
    gray = _np.dot(ar[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    mp = int(0.01*(y1-y0))
    coords = [
        (x, int(_np.median(_np.where(dark[:,x])[0])) + y0)
        for x in range(rw)
        if len(_np.where(dark[:,x])[0]) >= mp
    ]
    al = _np.array(left)
    grayl = _np.dot(al[y0:y1], [0.299, 0.587, 0.114])
    darkl = grayl < 200
    ys = _np.where(darkl[:, -2])[0]
    ymed = (int(_np.median(ys)) + y0) if ys.size >= mp else (y0+y1)//2
    full = [(0, ymed)] + coords

    blank = Image.new("RGB", (rw, rh), (255,255,255))
    d2 = ImageDraw.Draw(blank)
    if full:
        d2.line(full, fill=(0,0,255), width=3)

    # 6) reassemble & final crop
    combo = Image.new("RGB", (w, h))
    combo.paste(left.convert("RGB"), (0,0))
    combo.paste(blank, (x_split, 0))
    cx, cy = int(0.1*w), int(0.1*h)
    return combo.crop((cx, cy, w-cx, h-cy))
