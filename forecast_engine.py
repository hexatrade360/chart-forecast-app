
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, extract_split_point

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
    w, h = query_img.size
    split_x = extract_split_point(query_img)

    # DEBUG: Show split position
    debug_img = query_img.copy()
    draw_debug = ImageDraw.Draw(debug_img)
    draw_debug.line([(split_x, 0), (split_x, h)], fill=(255, 0, 0), width=2)
    steps.append((f"🔴 Detected Red Line @ x={split_x}", debug_img))

    for rank, idx in enumerate(top_idxs):
        name, score = names[idx], sims[idx]
        match_img = Image.open(os.path.join("data", "screenshots", name)).convert("RGB")
        steps.append((f"Match #{rank+1}: {name} (score={score:.4f})", match_img))

        qg = np.array(query_img.convert('L').resize((224,224))) / 255.0
        mg = np.array(match_img.convert('L').resize((224,224))) / 255.0
        corr = qg * mg
        fig, ax = plt.subplots()
        ax.imshow(corr, cmap='viridis', norm=Normalize(0,1))
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        heat = Image.open(buf).convert("RGB")
        plt.close(fig)
        steps.append((f"🔍 Correlation Heatmap #{rank+1}", heat))

        def prep(img):
            c = ImageEnhance.Contrast(img).enhance(1.5)
            g = cv2.cvtColor(np.array(c), cv2.COLOR_RGB2GRAY)
            return cv2.GaussianBlur(g, (3,3), 0)
        kp1, d1 = cv2.ORB_create(500).detectAndCompute(prep(query_img), None)
        kp2, d2 = cv2.ORB_create(500).detectAndCompute(prep(match_img), None)
        if d1 is not None and d2 is not None:
            matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, True).match(d1, d2), key=lambda x: x.distance)[:25]
            orb = cv2.drawMatches(np.array(query_img), kp1, np.array(match_img), kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            steps.append((f"🔗 ORB Matches #{rank+1}", Image.fromarray(orb)))

        if rank == 0:
            overlay = query_img.convert("RGBA")
            forecast_crop = match_img.resize((w, h)).crop((split_x, 0, w, h)).convert("RGBA")
            mask = forecast_crop.split()[-1].point(lambda p: 128)
            overlay.paste(forecast_crop, (split_x, 0), mask)
            ov_rgb = overlay.convert("RGB")
            final_overlay = ov_rgb
            steps.append(("📈 Final Forecast Overlay", ov_rgb))

    return (final_overlay, steps) if debug else final_overlay
