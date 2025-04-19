import os
import boto3
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from torchvision import models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AWS/S3 configuration from environment variables
AWS_REGION       = os.environ["AWS_REGION"]
S3_BUCKET_NAME   = os.environ["S3_BUCKET_NAME"]
S3_WEIGHTS_KEY   = os.environ["S3_WEIGHTS_KEY"]
WEIGHTS_PATH     = os.path.join("data", "model_weights.pth")

# Initialize S3 client
_s3_client = boto3.client("s3", region_name=AWS_REGION)

def download_weights_from_s3():
    """Download model weights from S3 if not already present locally."""
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if not os.path.isfile(WEIGHTS_PATH):
        print(f"📥 Downloading weights from s3://{S3_BUCKET_NAME}/{S3_WEIGHTS_KEY}")
        _s3_client.download_file(S3_BUCKET_NAME, S3_WEIGHTS_KEY, WEIGHTS_PATH)
        print(f"✅ Model weights saved to {WEIGHTS_PATH}")

# Helper to crop chart body
def crop_chart_body(img):
    """Crop 10% margins from each side to isolate the chart area."""
    w, h = img.size
    left, right = int(0.1 * w), int(0.9 * w)
    top, bottom = int(0.1 * h), int(0.9 * h)
    return img.crop((left, top, right, bottom))

# Define the embedding network to match training
class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 128)

    def forward(self, x):
        return self.base_model(x)

# Cache for loaded model
_model_cache = None

def load_model():
    """Load and cache the fine-tuned model weights."""
    global _model_cache
    if _model_cache is None:
        download_weights_from_s3()
        model = EmbeddingNet().to(device)
        raw_state = torch.load(WEIGHTS_PATH, map_location=device)
        # strip prefix if present
        new_state = {}
        for k, v in raw_state.items():
            name = k.split("embedding_net.")[-1]
            new_state[name] = v
        model.load_state_dict(new_state)
        model.eval()
        _model_cache = model
    return _model_cache

# Transformation matching training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img):
    """
    Compute embedding for a PIL image using the fine-tuned model,
    after cropping to the chart body.
    """
    # 1) Crop to focus on the chart
    chart_body = crop_chart_body(img)
    # 2) Transform to tensor
    tensor = transform(chart_body).unsqueeze(0).to(device)
    # 3) Forward through model
    model = load_model()
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb

def find_best_match(query_embedding, data_embeddings):
    """Find filename of the closest precomputed embedding."""
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb)
        if score < best_score:
            best_score = score
            best_match = fname
    return os.path.join("data", "screenshots", best_match)

def generate_overlay_forecast(query_img, match_img):
    """
    Generate forecast overlay by splitting and stitching according to red line.
    """
    w, h = query_img.size
    match_resized = match_img.resize((w, h))
    arr = np.array(match_resized)
    # red-line detection
    red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    prop = red_mask.sum(axis=0) / h
    cols = np.where(prop > 0.02)[0]
    if len(cols) >= 2:
        x_split = int(cols[-1])
    else:
        x_split = w // 2
    x_split = max(1, min(x_split, w-1))
    # crop halves
    left = query_img.crop((0, 0, x_split, h))
    right = match_resized.crop((x_split, 0, w, h))
    # continuity y on left
    y0, y1 = int(0.1 * h), int(0.9 * h)
    gray_l = np.dot(np.array(left)[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys_l = np.where(dark_l[:, -1])[0] if left.width > 0 else []
    y_med_l = int(np.median(ys_l)) + y0 if len(ys_l) > 0 else (y0 + y1) // 2
    # forecast coords on right
    gray_r = np.dot(np.array(right)[y0:y1], [0.299, 0.587, 0.114])
    dark_r = gray_r < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = []
    for x in range(right.width):
        ys = np.where(dark_r[:, x])[0]
        if len(ys) >= min_pix:
            coords.append((x, int(np.median(ys)) + y0))
    full_coords = [(0, y_med_l)] + coords
    # draw forecast line
    blank = Image.new("RGB", (right.width, h), (255, 255, 255))
    draw = ImageDraw.Draw(blank)
    if full_coords:
        draw.line(full_coords, fill=(0, 0, 255), width=3)
    # assemble and crop margins
    combined = Image.new("RGB", (w, h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(blank, (x_split, 0))
    cx, cy = int(0.1 * w), int(0.1 * h)
    return combined.crop((cx, cy, w - cx, h - cy))
