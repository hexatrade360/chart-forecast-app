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
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        _model_cache = model
    return _model_cache

# Transformer matching training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(img):
    """Compute embedding for a PIL image using the fine-tuned model."""
    model = load_model()
    tensor = transform(img).unsqueeze(0).to(device)
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
    # Return full path to the matched screenshot
    return os.path.join("data", "screenshots", best_match)

def generate_overlay_forecast(query_img, match_img):
    """Generate forecast overlay by splitting and stitching according to red line."""
    # Ensure query and match same size
    w, h = query_img.size
    match_resized = match_img.resize((w, h))
    match_arr = np.array(match_resized)

    # Detect red vertical line (forecast split)
    red_mask = (match_arr[:,:,0] > 200) & (match_arr[:,:,1] < 80) & (match_arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]

    if len(red_cols) >= 2:
        x_split = int(red_cols[-1])
    else:
        x_split = w // 2

    # Clamp split to valid range
    x_split = max(1, min(x_split, w-1))

    # Crop left and right regions
    left_region  = query_img.crop((0, 0, x_split, h))
    right_region = match_resized.crop((x_split, 0, w, h))

    # Determine continuity point on left
    y0, y1 = int(0.10 * h), int(0.90 * h)
    arr_left = np.array(left_region)
    gray_l = np.dot(arr_left[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    ys_l = np.where(dark_l[:, -1])[0] if left_region.width > 0 else []
    y_med_l = int(np.median(ys_l)) + y0 if len(ys_l) > 0 else (y0 + y1) // 2

    # Trace forecast line on right
    arr_right = np.array(right_region)
    gray = np.dot(arr_right[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = []
    for x in range(right_region.width):
        ys = np.where(dark[:, x])[0]
        if len(ys) >= min_pix:
            coords.append((x, int(np.median(ys)) + y0))
    full_coords = [(0, y_med_l)] + coords

    # Draw on blank canvas
    blank_right = Image.new('RGB', (right_region.width, h), (255, 255, 255))
    draw = ImageDraw.Draw(blank_right)
    if full_coords:
        draw.line(full_coords, fill=(0, 0, 255), width=3)

    # Combine and crop margins
    combined = Image.new('RGB', (w, h), (255,255,255))
    combined.paste(left_region, (0,0))
    combined.paste(blank_right, (x_split,0))
    crop_x = int(0.10 * w)
    crop_y = int(0.10 * h)
    cropped = combined.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    return cropped
