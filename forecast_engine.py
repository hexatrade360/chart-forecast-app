# ✅ Version: manual-red-line v1.1 — stable forecast overlay with manual red line support

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding

def process_forecast_pipeline(query_img, top_k=3, debug=False):
    emb_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs = np.stack([data_embeddings[n].numpy() for n in names])

    q = extract_embedding(query_img).numpy()[None, :]
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    steps = []
    final_overlay = None

    for rank, idx in enumerate(top_idxs):
        name, score = names[idx], sims[idx]
        match_img = Image.open(os.path.join("data", "screenshots", name)).convert("RGB")
        steps.append((f"Match #{rank+1}: {name} (score={score:.4f})", match_img))

        # Skip drawing red line if it already exists
        arr = np.array(query_img)
        red_mask = (arr[:,:,0] > 200) & (arr[:,:,1] < 80) & (arr[:,:,2] < 80)
        red_cols = np.where(red_mask.sum(axis=0) > 0.02 * arr.shape[0])[0]
        if len(red_cols) > 0:
            split = red_cols[-1]
        else:
            split = query_img.size[0] // 2

        forecast_crop = match_img.resize(query_img.size).crop((split, 0, query_img.size[0], query_img.size[1])).convert("RGBA")
        overlay = query_img.convert("RGBA")
        mask = forecast_crop.split()[-1].point(lambda p: 128)
        overlay.paste(forecast_crop, (split, 0), mask)

        steps.append((f"📈 Overlay #{rank+1}", overlay.convert("RGB")))

        if rank == 0:
            # Extract blue forecast line
            left = overlay.crop((0, 0, split, query_img.size[1]))
            right = overlay.crop((split, 0, query_img.size[0], query_img.size[1]))
            rh, rw = right.size[1], right.size[0]
            arr_r = np.array(right)
            y0, y1 = int(0.1*rh), int(0.9*rh)
            gray = np.dot(arr_r[y0:y1], [0.299,0.587,0.114])
            dark = gray < 200
            minp = int(0.01 * (y1 - y0))
            coords = [(x, int(np.median(np.where(dark[:,x])[0])) + y0)
                      for x in range(rw) if len(np.where(dark[:,x])[0]) >= minp]

            arr_l = np.array(left)
            grayl = np.dot(arr_l[y0:y1], [0.299,0.587,0.114])
            darkl = grayl < 200
            ys = np.where(darkl[:,-2])[0]
            y_med = int(np.median(ys)) + y0 if len(ys) >= minp else int((y0 + y1) / 2)
            full = [(0, y_med)] + coords

            blank = Image.new("RGB", (rw, rh), (255,255,255))
            draw2 = ImageDraw.Draw(blank)
            if full:
                draw2.line(full, fill=(0, 0, 255), width=3)

            combo = Image.new("RGB", query_img.size)
            combo.paste(left, (0, 0))
            combo.paste(blank, (split, 0))

            cx, cy = int(0.1 * query_img.size[0]), int(0.1 * query_img.size[1])
            final = combo.crop((cx, cy, query_img.size[0]-cx, query_img.size[1]-cy))
            final_overlay = final
            steps.append(("🔵 Forecast Line Only", blank))
            steps.append(("Final Cropped Output", final))

    if debug:
        return final_overlay, steps
    return final_overlay
