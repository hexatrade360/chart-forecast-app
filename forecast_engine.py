# forecast_engine.py
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import extract_embedding

def process_forecast_pipeline(query_img, top_k=3, debug=False):
    """
    If debug=True, returns (final_img, steps)
      where steps = [ (label, PIL.Image), ... ]
    else returns final_img.
    """
    # load embeddings
    emb_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs  = np.stack([data_embeddings[n].numpy() for n in names])

    # embed query
    q = extract_embedding(query_img).numpy()[None,:]
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    steps = []
    best_overlay = None

    for rank, idx in enumerate(top_idxs):
        name, score = names[idx], sims[idx]
        # 1) Load images
        match_img = Image.open(os.path.join("data","screenshots", name)).convert("RGB")
        steps.append((f"Match #{rank+1}: {name} (score={score:.4f})", match_img))

        # 2) Correlation heatmap
        q_g  = np.array(query_img.convert("L").resize((224,224))) / 255.0
        m_g  = np.array(match_img.convert("L").resize((224,224))) / 255.0
        corr = q_g * m_g
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="viridis", norm=Normalize(0,1))
        ax.axis("off")
        fig.canvas.draw()
        heatsrc = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        heatsrc = heatsrc.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        steps.append((f"🔍 Correlation Heatmap #{rank+1}", Image.fromarray(heatsrc)))

        # 3) ORB matches
        # a) contrast-enhance & gray
        def prep(img):
            c = ImageEnhance.Contrast(img).enhance(1.5)
            g = cv2.cvtColor(np.array(c), cv2.COLOR_RGB2GRAY)
            return cv2.GaussianBlur(g, (3,3), 0)
        kp1, d1 = cv2.ORB_create(500).detectAndCompute(prep(query_img), None)
        kp2, d2 = cv2.ORB_create(500).detectAndCompute(prep(match_img), None)
        if d1 is not None and d2 is not None:
            matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, True)
                             .match(d1,d2), key=lambda x: x.distance)[:25]
            orb_img = cv2.drawMatches(np.array(query_img), kp1,
                                      np.array(match_img), kp2,
                                      matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            steps.append((f"🔗 ORB Matches #{rank+1}", Image.fromarray(orb_img)))

        # 4) Build overlay + split red line
        w, h = match_img.size
        mid = w//2
        forecast_crop = match_img.crop((mid,0,w,h)).convert("RGBA")
        overlay = query_img.convert("RGBA")
        mask = forecast_crop.split()[-1].point(lambda p:128)
        overlay.paste(forecast_crop, (mid,0), mask)
        d = ImageDraw.Draw(overlay)
        d.line([(mid,0),(mid,h)], fill="red", width=2)
        steps.append((f"📈 Overlay #{rank+1}", overlay.convert("RGB")))

        if rank==0:
            best_overlay = overlay.convert("RGB")

    if debug:
        return best_overlay, steps
    else:
        return best_overlay
