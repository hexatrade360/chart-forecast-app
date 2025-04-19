import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding

def process_forecast_pipeline(query_img):
    """
    1) Load embeddings (filename -> tensor)
    2) Compute query embedding
    3) Find best match via cosine similarity
    4) Load match image, build RGBA overlay of forecast half
    5) Draw red split line
    6) Crop margins and return final image
    """
    # 1) Load precomputed embeddings
    emb_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(emb_path)
    filenames = list(data_embeddings.keys())
    vectors = np.stack([data_embeddings[f].numpy() for f in filenames])

    # 2) Compute query embedding
    q_emb = extract_embedding(query_img).numpy()[None, :]

    # 3) Cosine similarity to find best match
    sims = cosine_similarity(q_emb, vectors)[0]
    best_idx = int(np.argmax(sims))
    best_name, best_score = filenames[best_idx], sims[best_idx]
    print(f"🤝 Best match: {best_name} (cosine={best_score:.4f})")

    # 4) Load the matching screenshot
    match_img = Image.open(os.path.join("data", "screenshots", best_name)).convert("RGB")
    w, h = match_img.size
    mid = w // 2

    # 5) Create forecast overlay (right half) with constant alpha
    forecast_crop = match_img.crop((mid, 0, w, h)).convert("RGBA")
    overlay = query_img.convert("RGBA")
    mask = Image.new("L", (w - mid, h), 128)
    overlay.paste(forecast_crop, (mid, 0), mask)

    # 6) Draw the red split line
    draw = ImageDraw.Draw(overlay)
    draw.line([(mid, 0), (mid, h)], fill="red", width=2)

    # 7) Crop 10% margins
    cx, cy = int(0.1 * w), int(0.1 * h)
    final_img = overlay.crop((cx, cy, w - cx, h - cy))
    return final_img
