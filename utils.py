import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from torchvision import models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AWS/S3 configuration
AWS_REGION       = os.environ["AWS_REGION"]
S3_BUCKET_NAME   = os.environ["S3_BUCKET_NAME"]
S3_WEIGHTS_KEY   = os.environ["S3_WEIGHTS_KEY"]
WEIGHTS_PATH     = os.path.join("data", "model_weights.pth")

_s3_client = boto3.client("s3", region_name=AWS_REGION)

def download_weights_from_s3():
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if not os.path.isfile(WEIGHTS_PATH):
        _s3_client.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)

def crop_chart_body(img):
    w,h = img.size
    left, right = int(0.1*w), int(0.9*w)
    top, bottom = int(0.1*h), int(0.9*h)
    return img.crop((left, top, right, bottom))

class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 128)
    def forward(self,x):
        return self.base_model(x)

_model_cache = None
def load_model():
    global _model_cache
    if _model_cache is None:
        download_weights_from_s3()
        model = EmbeddingNet().to(device)
        raw = torch.load(WEIGHTS_PATH, map_location=device)
        state = {k.split("embedding_net.")[-1]:v for k,v in raw.items()}
        model.load_state_dict(state)
        model.eval()
        _model_cache = model
    return _model_cache

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

def extract_embedding(img):
    body = crop_chart_body(img)
    tensor = transform(body).unsqueeze(0).to(device)
    with torch.no_grad():
        return load_model()(tensor).squeeze().cpu()

def find_best_match(query_embedding, data_embeddings):
    best, f = None, float("inf")
    for fname, emb in data_embeddings.items():
        d = torch.dist(query_embedding, emb).item()
        if d < f:
            f, best = d, fname
    return os.path.join("data","screenshots", best)

def generate_overlay_forecast(query_img, match_img):
    w, h = query_img.size
    mid = w // 2

    # Left and right halves
    left = query_img.crop((0, 0, mid, h))
    right_full = match_img.resize((w, h))
    right = right_full.crop((mid, 0, w, h))

    # Trace forecast line on right
    rh, rw = right.size[1], right.size[0]
    arr_r = np.array(right)
    y0, y1 = int(0.10 * rh), int(0.90 * rh)
    gray = np.dot(arr_r[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = [(x, int(np.median(np.where(dark[:,x])[0])) + y0)
              for x in range(rw) if len(np.where(dark[:,x])[0]) >= min_pix]

    # Continuity point from left region
    arr_l = np.array(left)
    gray_l = np.dot(arr_l[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys_l = np.where(dark_l[:, -1])[0]
    if len(ys_l) >= min_pix:
        y0l = int(np.median(ys_l)) + y0
    else:
        y0l = (y0 + y1) // 2

    full_coords = [(0, y0l)] + coords

    # Smooth the y-coordinates with moving average
    xs, ys = zip(*full_coords)
    window = 5
    ys_sm = np.convolve(ys, np.ones(window)/window, mode="same").astype(int)
    full_coords = list(zip(xs, ys_sm))

    # Draw blue forecast line on blank right canvas
    blank = Image.new("RGB", (rw, rh), (255, 255, 255))
    draw = ImageDraw.Draw(blank)
    if full_coords:
        draw.line(full_coords, fill=(0, 0, 255), width=3)

    # Combine and draw red split line
    combo = Image.new("RGB", (w, h), (255, 255, 255))
    combo.paste(left, (0, 0))
    combo.paste(blank, (mid, 0))
    draw2 = ImageDraw.Draw(combo)
    draw2.line([(mid, 0), (mid, h)], fill=(255, 0, 0), width=2)

    # Crop margins
    cx, cy = int(0.1 * w), int(0.1 * h)
    return combo.crop((cx, cy, w - cx, h - cy))
