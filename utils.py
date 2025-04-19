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

def crop_chart_body(img):
    """Crop 10% margins from each side to isolate the chart area."""
    w, h = img.size
    left, right = int(0.1 * w), int(0.9 * w)
    top, bottom = int(0.1 * h), int(0.9 * h)
    return img.crop((left, top, right, bottom))

class EmbeddingNet(torch.nn.Module):
    """ResNet18-based embedding network matching training architecture."""
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 128)

    def forward(self, x):
        return self.base_model(x)

_model_cache = None
def load_model():
    """Load and cache fine-tuned model weights."""
    global _model_cache
    if _model_cache is None:
        download_weights_from_s3()
        model = EmbeddingNet().to(device)
        raw_state = torch.load(WEIGHTS_PATH, map_location=device)
        # Strip any 'embedding_net.' prefix
        new_state = {k.split("embedding_net.")[-1]: v for k, v in raw_state.items()}
        model.load_state_dict(new_state)
        model.eval()
        _model_cache = model
    return _model_cache

# Image transform matching training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img):
    """
    Compute embedding for a PIL image using the fine-tuned model,
    cropping to the chart body first.
    """
    chart_body = crop_chart_body(img)
    tensor = transform(chart_body).unsqueeze(0).to(device)
    model = load_model()
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu()
    return emb

def find_best_match(query_embedding, data_embeddings):
    """
    Find filename of the closest precomputed embedding using Euclidean distance.
    """
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb).item()
        if score < best_score:
            best_score = score
            best_match = fname
    return os.path.join("data", "screenshots", best_match)

def generate_overlay_forecast(query_img, match_img):
    """
    Generate the final forecast overlay:
    - Detect red split line
    - Split left (query) and right (forecast) regions
    - Trace forecast candles into a blue line
    - Paste blue line on white canvas
    - Draw red vertical split line
    - Crop 10% margins
    """
    # Ensure same size
    w, h = query_img.size
    match_r = match_img.resize((w, h))
    arr = np.array(match_r)

    # 1️⃣ Detect red vertical split
    red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    if len(red_cols) >= 2:
        x_split = int(red_cols[-1])
    else:
        x_split = int(np.argmax(prop_red))
    x_split = max(1, min(x_split, w-1))

    # 2️⃣ Separate regions
    left_region = query_img.crop((0, 0, x_split+1, h))
    right_region = match_r.crop((x_split+1, 0, w, h))

    # 3️⃣ Trace forecast line on right
    rh, rw = right_region.size[1], right_region.size[0]
    arr_r = np.array(right_region)
    y0, y1 = int(0.10 * rh), int(0.90 * rh)
    gray = np.dot(arr_r[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = [(x, int(np.median(np.where(dark[:,x])[0])) + y0)
              for x in range(rw) if len(np.where(dark[:,x])[0]) >= min_pix]

    # 4️⃣ Continuity point from left region
    arr_l = np.array(left_region)
    gray_l = np.dot(arr_l[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    col_idx = left_region.size[0] - 2
    ys_l = np.where(dark_l[:, col_idx])[0]
    if len(ys_l) >= min_pix:
        y_med_l = int(np.median(ys_l)) + y0
    else:
        y_med_l = int((y0 + y1) / 2)
    full_coords = [(0, y_med_l)] + coords

    blank_right = Image.new('RGB', (rw, rh), (255, 255, 255))
    draw = ImageDraw.Draw(blank_right)
    if full_coords:
        draw.line(full_coords, fill=(0, 0, 255), width=3)

    # 5️⃣ Reassemble and draw red split line
    combined = Image.new('RGB', (w, h), (255, 255, 255))
    combined.paste(left_region, (0, 0))
    combined.paste(blank_right, (x_split+1, 0))
    draw2 = ImageDraw.Draw(combined)
    draw2.line([(x_split, 0), (x_split, h)], fill=(255, 0, 0), width=2)

    # 6️⃣ Crop margins
    cx, cy = int(0.1 * w), int(0.1 * h)
    final_img = combined.crop((cx, cy, w - cx, h - cy))
    return final_img
