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
    """Crop, transform, and run through embedding net."""
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = load_model()(tensor).squeeze().cpu()
    return emb

def generate_overlay_forecast(query_img: Image.Image, match_img: Image.Image) -> Image.Image:
    """
    1) Dynamically detect where the colored candles end in query_img,
    2) Paste the forecast from match_img to its right,
    3) Draw red split line,
    4) Trace forecast candles into a blue line,
    5) Crop 10% margins.
    """
    # — detect split column on query
    arr_q = np.array(query_img.convert("RGB"))
    gray_q = np.dot(arr_q[...,:3], [0.299, 0.587, 0.114])
    h, w = gray_q.shape
    mask = gray_q < 250
    counts = mask.sum(axis=0)
    minpix = int(0.01 * h)
    valid = counts[: int(0.8 * w)]
    cols = np.where(valid > minpix)[0]
    x_split = int(cols[-1] if len(cols) else w // 2)

    # — build RGBA overlay with forecast pasted
    overlay = query_img.convert("RGBA")
    m = match_img.resize((w, h)).convert("RGBA")
    fc = m.crop((x_split, 0, w, h))
    mask_fc = fc.split()[-1].point(lambda p: 128)
    overlay.paste(fc, (x_split, 0), mask_fc)

    # — draw red split line
    draw = ImageDraw.Draw(overlay)
    draw.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # — trace blue forecast line
    left  = overlay.crop((0, 0, x_split, h))
    right = overlay.crop((x_split, 0, w, h))
    arr_r = np.array(right)
    rh, rw = arr_r.shape[:2]
    y0, y1 = int(0.1 * rh), int(0.9 * rh)
    gray_r = np.dot(arr_r[y0:y1], [0.299, 0.587, 0.114])
    dark_r = gray_r < 200
    minp = int(0.01 * (y1 - y0))

    coords = [
        (x, int(np.median(np.where(dark_r[:, x])[0])) + y0)
        for x in range(rw)
        if len(np.where(dark_r[:, x])[0]) >= minp
    ]

    arr_l = np.array(left)
    gray_l = np.dot(arr_l[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys = np.where(dark_l[:, -2])[0]
    y_med = int(np.median(ys)) + y0 if len(ys) >= minp else (y0 + y1) // 2
    full_coords = [(0, y_med)] + coords

    blank = Image.new("RGB", (rw, rh), (255, 255, 255))
    d2 = ImageDraw.Draw(blank)
    if full_coords:
        d2.line(full_coords, fill=(0, 0, 255), width=3)

    # — reassemble & crop margins
    combined = Image.new("RGB", (w, h))
    combined.paste(left.convert("RGB"), (0, 0))
    combined.paste(blank, (x_split, 0))
    cx, cy = int(0.1 * w), int(0.1 * h)
    return combined.crop((cx, cy, w - cx, h - cy))
