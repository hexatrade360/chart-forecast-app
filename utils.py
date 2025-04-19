import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

# ===============================
# 📦 Load model with correct architecture
# ===============================
class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.convnet(x).view(x.size(0), -1)
        return self.fc(x)

def load_model(weights_path="data/model_weights.pth"):
    model = EmbeddingNet()
    state = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

# ===============================
# 🧠 Extract embedding from image
# ===============================
def extract_embedding(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    tensor = transform(image_pil.convert("RGB")).unsqueeze(0)
    return load_model()(tensor).squeeze().cpu()

# ===============================
# 🔍 Find best match from embeddings
# ===============================
def find_best_match(query_vec, data_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    names = list(data_embeddings.keys())
    vecs = torch.stack([data_embeddings[k] for k in names])
    sims = cosine_similarity(query_vec.unsqueeze(0).numpy(), vecs.numpy())[0]
    best_idx = sims.argmax()
    return names[best_idx], sims[best_idx]

# ===============================
# 🔥 Generate overlay with diagnostics + forecast line
# ===============================
def generate_overlay_forecast(query_img, match_img):
    w, h = match_img.size
    mid = w // 2

    # Overlay forecast area from matched image onto query
    forecast_crop = match_img.crop((mid, 0, w, h)).convert("RGBA")
    overlay = query_img.convert("RGBA")
    mask = forecast_crop.split()[-1].point(lambda p: 128)
    overlay.paste(forecast_crop, (mid, 0), mask)

    # Draw red divider
    draw = ImageDraw.Draw(overlay)
    draw.line([(mid, 0), (mid, h)], fill="red", width=2)

    # ====================
    # 🔵 Draw Blue Forecast Line
    # ====================
    arr = np.array(overlay)
    red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
    prop_red = red_mask.sum(axis=0) / h
    red_cols = np.where(prop_red > 0.02)[0]
    x_split = red_cols[-1] if len(red_cols) >= 2 else int(np.argmax(prop_red))

    left_region = overlay.crop((0, 0, x_split+1, h))
    right_region = overlay.crop((x_split+1, 0, w, h))
    rh, rw = right_region.size[1], right_region.size[0]

    y0, y1 = int(0.10 * rh), int(0.90 * rh)
    arr_right = np.array(right_region)
    gray = np.dot(arr_right[y0:y1], [0.299, 0.587, 0.114])
    dark = gray < 200
    min_pix = int(0.01 * (y1 - y0))
    coords = [(x, int(np.median(np.where(dark[:, x])[0])) + y0)
              for x in range(rw) if len(np.where(dark[:, x])[0]) >= min_pix]

    arr_left = np.array(left_region)
    gray_l = np.dot(arr_left[y0:y1], [0.299, 0.587, 0.114])
    dark_l = gray_l < 200
    col_idx = left_region.size[0] - 2
    ys_l = np.where(dark_l[:, col_idx])[0]
    y_med_l = int(np.median(ys_l)) + y0 if len(ys_l) >= min_pix else int((y0 + y1) / 2)
    full_coords = [(0, y_med_l)] + coords

    blank_right = Image.new('RGB', (rw, rh), (255, 255, 255))
    draw = ImageDraw.Draw(blank_right)
    draw.line(full_coords, fill=(0, 0, 255), width=3)

    combined = Image.new('RGB', (w, h))
    combined.paste(left_region.convert("RGB"), (0, 0))
    combined.paste(blank_right, (x_split + 1, 0))

    crop_x = int(0.10 * w)
    crop_y = int(0.10 * h)
    cropped = combined.crop((crop_x, crop_y, w - crop_x, h - crop_y))

    return cropped
