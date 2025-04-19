
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from utils import extract_embedding

def process_forecast_pipeline(query_img, top_k=3, debug=False):
    emb_path = os.path.join("data","embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs = np.stack([data_embeddings[n].detach().numpy() for n in names])

    q = extract_embedding(query_img).detach().numpy()[None, :]
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    steps = []
    final_overlay = None

    for rank, idx in enumerate(top_idxs):
        name = names[idx]
        match_img = Image.open(os.path.join("data/screenshots", name)).convert("RGB")
        steps.append((f"Match #{rank+1}: {name} (score={sims[idx]:.4f})", match_img))

        # Red line detection on query
        arr = np.array(query_img)
        h, w = arr.shape[:2]
        red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
        prop_red = red_mask.sum(axis=0) / h
        red_cols = np.where(prop_red > 0.02)[0]
        if len(red_cols) >= 2:
            x_split = red_cols[-1]
        else:
            x_split = w // 2

        # Split regions
        left_region = query_img.crop((0, 0, x_split + 1, h))
        right_region = match_img.crop((x_split + 1, 0, w, h)).resize((w - x_split - 1, h))

        # Trace blue forecast
        arr_right = np.array(right_region)
        y0, y1 = int(0.10 * h), int(0.90 * h)
        gray = np.dot(arr_right[y0:y1], [0.299, 0.587, 0.114])
        dark = gray < 200
        min_pix = int(0.01 * (y1 - y0))
        coords = [(x, int(np.median(np.where(dark[:, x])[0])) + y0)
                  for x in range(arr_right.shape[1]) if len(np.where(dark[:, x])[0]) >= min_pix]

        # Extend line from last candle on left
        arr_left = np.array(left_region)
        gray_l = np.dot(arr_left[y0:y1], [0.299, 0.587, 0.114])
        dark_l = gray_l < 200
        col_idx = arr_left.shape[1] - 2
        ys_l = np.where(dark_l[:, col_idx])[0]
        y_med_l = int(np.median(ys_l)) + y0 if len(ys_l) >= min_pix else int((y0 + y1) / 2)
        full_coords = [(0, y_med_l)] + coords

        # Draw blue line forecast
        blank_right = Image.new("RGB", right_region.size, (255, 255, 255))
        draw = ImageDraw.Draw(blank_right)
        if full_coords:
            draw.line(full_coords, fill=(0, 0, 255), width=3)

        # Merge and crop
        merged = Image.new("RGB", (w, h))
        merged.paste(left_region, (0, 0))
        merged.paste(blank_right, (x_split + 1, 0))
        cx, cy = int(0.10 * w), int(0.10 * h)
        final = merged.crop((cx, cy, w - cx, h - cy))
        final_overlay = final

        if debug:
            steps.append(("🔵 Forecast Line Only", blank_right))
            steps.append(("Final Cropped Output", final))

    if debug:
        return final_overlay, steps
    return final_overlay
