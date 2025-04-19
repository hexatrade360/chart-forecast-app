import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ─── Configuration ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join("data", "model_weights.pth")

AWS_REGION     = os.environ.get("AWS_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_WEIGHTS_KEY = os.environ.get("S3_WEIGHTS_KEY")

_s3 = boto3.client("s3", region_name=AWS_REGION) if (AWS_REGION and S3_BUCKET_NAME and S3_WEIGHTS_KEY) else None

# ─── Model Definition ──────────────────────────────────────────────────────────
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
        # download if missing
        if _s3 and not os.path.isfile(WEIGHTS_PATH):
            os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
            _s3.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)
        model = ChartEmbeddingNet().to(device)
        state = torch.load(WEIGHTS_PATH, map_location=device)
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
    w,h = img.size
    body = img.crop((int(0.1*w), int(0.1*h), int(0.9*w), int(0.9*h)))
    tensor = _transform(body).unsqueeze(0).to(device)
    with torch.no_grad():
        return load_model()(tensor).squeeze().cpu()

# ─── Overlay + Forecast Line ───────────────────────────────────────────────────
def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    """
    1) Detect all colored‑candle columns in query.
    2) If any exist, find first blank run (>=2cols) and set x_split there;
       otherwise fallback to center.
    3) Paste forecast, draw red divider at x_split, trace blue line, then crop.
    """
    import numpy as _np

    w, h = query_img.size

    # 1) find candle columns: any channel <240
    arr = _np.array(query_img.convert("RGB"))
    candle_mask = _np.any(arr < 240, axis=2)
    counts = candle_mask.sum(axis=0)
    minpix = int(0.01 * h)
    candle_cols = _np.where(counts > minpix)[0]

    # 2) determine split
    if candle_cols.size:
        # count non‑candle fraction
        non_frac = (h - counts) / h
        blank = _np.where(non_frac > 0.99)[0]
        x_split = w // 2
        if blank.size:
            # find runs of >=2 consecutive blank cols
            groups = _np.split(blank, _np.where(_np.diff(blank)!=1)[0]+1)
            for g in groups:
                if g.size >= 2:
                    x_split = int(g[0])
                    break
            else:
                x_split = int(blank[0])
    else:
        x_split = w // 2

    # 3) overlay setup
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w,h)).convert("RGBA")

    # 4) paste forecast region
    fc = m.crop((x_split,0,w,h))
    mask_fc = fc.split()[-1].point(lambda p:128)
    overlay.paste(fc, (x_split,0), mask_fc)

    # 5) draw red divider
    draw = ImageDraw.Draw(overlay)
    draw.line([(x_split,0),(x_split,h)], fill=(255,0,0), width=2)

    # 6) trace blue line on forecast
    left  = overlay.crop((0,0,x_split,h))
    right = overlay.crop((x_split,0,w,h))
    ar = _np.array(right)
    rh,rw = ar.shape[:2]
    y0,y1 = int(0.1*rh), int(0.9*rh)
    gray = _np.dot(ar[y0:y1], [0.299,0.587,0.114])
    dark = gray < 200
    mp = int(0.01*(y1-y0))
    coords = [
        (x, int(_np.median(_np.where(dark[:,x])[0]))+y0)
        for x in range(rw)
        if len(_np.where(dark[:,x])[0]) >= mp
    ]
    al = _np.array(left)
    grayl = _np.dot(al[y0:y1], [0.299,0.587,0.114])
    darkl = grayl < 200
    ys = _np.where(darkl[:,-2])[0]
    ymed = int(_np.median(ys))+y0 if ys.size>=mp else (y0+y1)//2
    full = [(0,ymed)] + coords

    blank = Image.new("RGB",(rw,rh),(255,255,255))
    d2 = ImageDraw.Draw(blank)
    if full:
        d2.line(full, fill=(0,0,255), width=3)

    # 7) reassemble + final crop
    combo = Image.new("RGB",(w,h))
    combo.paste(left.convert("RGB"),(0,0))
    combo.paste(blank,(x_split,0))
    cx,cy = int(0.1*w), int(0.1*h)
    return combo.crop((cx, cy, w-cx, h-cy))
