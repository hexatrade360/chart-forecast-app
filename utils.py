import numpy as np
from PIL import Image, ImageDraw
import torch
import os
import torchvision.transforms as transforms
from torchvision import models

# 🔧 Same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🧠 Load the same ResNet18 structure used during training
class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 128)

    def forward(self, x):
        return self.base_model(x)

# Cache model
_model_cache = None
def load_model():
    global _model_cache
    if _model_cache is None:
        model = EmbeddingNet()
        model.eval()
        _model_cache = model
    return _model_cache

# ✅ Updated embedding extraction
def extract_embedding(img):
    model = load_model()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_tensor).squeeze()
    return emb

# 🔍 Match closest screenshot
def find_best_match(query_embedding, data_embeddings):
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb)
        if score < best_score:
            best_score = score
            best_match = fname
    # Return full path here for simplicity
    return os.path.join("data", "screenshots", best_match)

# 🖼 Generate overlay image based on original notebook logic
def generate_overlay_forecast(query_img, match_img):
    # ensure same size
    w, h = query_img.size
    match_arr = np.array(match_img.resize((w, h)))
    # detect split x on match image via red vertical line
    red_mask = (match_arr[:,:,0] > 200) & (match_arr[:,:,1] < 80) & (match_arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    if len(red_cols) >= 2:
        x_split = int(red_cols[-1])
    else:
        x_split = int(np.argmax(prop_red))
    # crop regions
    left_region = query_img.crop((0, 0, x_split, h))
    right_region = match_img.crop((x_split, 0, w, h))
    # prepare forecast line blank canvas
    rw, rh = right_region.size
    arr_right = np.array(right_region)
    y0, y1 = int(0.10 * rh), int(0.90 * rh)
    gray = np.dot(arr_right[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = []
    for x in range(rw):
        ys = np.where(dark[:, x])[0]
        if len(ys) >= min_pix:
            coords.append((x, int(np.median(ys)) + y0))
    # connect from left boundary
    arr_left = np.array(left_region)
    gray_l = np.dot(arr_left[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    col_idx = left_region.size[0] - 1
    ys_l = np.where(dark_l[:, col_idx])[0]
    if len(ys_l) >= min_pix:
        y_med_l = int(np.median(ys_l)) + y0
    else:
        y_med_l = int((y0 + y1) / 2)
    full_coords = [(0, y_med_l)] + coords
    blank_right = Image.new('RGB', (rw, rh), (255, 255, 255))
    draw = ImageDraw.Draw(blank_right)
    draw.line(full_coords, fill=(0, 0, 255), width=3)
    # reassemble
    combined = Image.new('RGB', (w, h), (255,255,255))
    combined.paste(left_region, (0,0))
    combined.paste(blank_right, (x_split,0))
    # crop padding
    crop_x = int(0.10 * w)
    crop_y = int(0.10 * h)
    cropped = combined.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    return cropped
