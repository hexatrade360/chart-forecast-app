
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding

def process_forecast_pipeline(query_img, top_k=3, debug=False):
    emb_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs = np.stack([data_embeddings[n].detach().numpy() for n in names])
    
    q = extract_embedding(query_img).detach().numpy()[None, :]
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    best_overlay = None
    steps = []

    for rank, idx in enumerate(top_idxs):
        name = names[idx]
        match_img = Image.open(os.path.join("data", "screenshots", name)).convert("RGB")
        match_img = match_img.resize(query_img.size)

        if rank == 0:
            arr = np.array(query_img)
            h, w = arr.shape[:2]

            # Detect manual red line
            red_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
            red_column_ratios = red_mask.sum(axis=0) / h
            red_columns = np.where(red_column_ratios > 0.5)[0]
            if len(red_columns) == 0:
                raise ValueError("❌ No red vertical line found in query image.")
            split_x = red_columns[0]

            # Split
            left_region = query_img.crop((0, 0, split_x, h))
            right_region = match_img.crop((split_x, 0, w, h))

            # Trace blue line
            arr_right = np.array(right_region)
            y0, y1 = int(0.1 * h), int(0.9 * h)
            gray = np.dot(arr_right[y0:y1], [0.299, 0.587, 0.114])
            dark = gray < 200
            min_pix = int(0.01 * (y1 - y0))
            coords = [(x, int(np.median(np.where(dark[:, x])[0])) + y0)
                      for x in range(arr_right.shape[1])
                      if len(np.where(dark[:, x])[0]) >= min_pix]

            arr_left = np.array(left_region)
            gray_l = np.dot(arr_left[y0:y1], [0.299, 0.587, 0.114])
            dark_l = gray_l < 200
            col_idx = arr_left.shape[1] - 2
            ys_l = np.where(dark_l[:, col_idx])[0]
            y_med_l = int(np.median(ys_l)) + y0 if len(ys_l) >= min_pix else int((y0 + y1) / 2)

            full_coords = [(0, y_med_l)] + coords
            blank_right = Image.new("RGB", (right_region.size), (255, 255, 255))
            draw = ImageDraw.Draw(blank_right)
            draw.line(full_coords, fill=(0, 0, 255), width=3)

            combined = Image.new("RGB", (w, h))
            combined.paste(left_region, (0, 0))
            combined.paste(blank_right, (split_x, 0))

            crop_x = int(0.10 * w)
            crop_y = int(0.10 * h)
            final = combined.crop((crop_x, crop_y, w - crop_x, h - crop_y))

            steps.append(("Final Cropped Output", final))
            best_overlay = final

    if debug:
        return best_overlay, steps
    return best_overlay
