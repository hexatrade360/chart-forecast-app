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

def _find_split_column(img: Image.Image) -> int:
    """
    Scan each column of img; find the first column with zero
    'candle' pixels (any channel < 240). Fallback to center.
    """
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < 240, axis=2)      # True where there's a candle pixel
    col_counts = mask.sum(axis=0)
    blanks = np.where(col_counts == 0)[0]
    if blanks.size:
        return int(blanks[0])
    return img.width // 2

def process_forecast_pipeline(query_img: Image.Image, top_k=3, debug=False):
    """
    1) Load embeddings & find Top‑K matches.
    2) For each: correlation heatmap + ORB match visualization.
    3) Initial overlay with red divider at dynamic split.
    4) For best match: trace blue forecast line & crop margins.
    """
    # ─── Load & prepare embeddings ───────────────────────────────────────────────
    emb_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs  = np.stack([data_embeddings[n].numpy() for n in names])

    # ─── Compute similarities ─────────────────────────────────────────────────────
    q = extract_embedding(query_img).numpy()[None, :]
    import numpy as np
    qv = q.numpy().flatten()
    print("▶ QUERY EMBEDDING FIRST 5:", qv[:5])
    print("▶ QUERY EMBEDDING NORM:", np.linalg.norm(qv))
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    steps = []
    final_overlay = None

    for rank, idx in enumerate(top_idxs):
        name, score = names[idx], sims[idx]
        match_path = os.path.join("data", "screenshots", name)
        match_img  = Image.open(match_path).convert("RGB")

        # 1️⃣ Record which file and score
        steps.append((f"Match #{rank+1}: {name} (score={score:.4f})", match_img))

        # 2️⃣ Correlation heatmap
        qg = np.array(query_img.convert("L").resize((224,224))) / 255.0
        mg = np.array(match_img.convert("L").resize((224,224))) / 255.0
        corr = qg * mg
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="viridis", norm=Normalize(0,1))
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        heat = Image.open(buf).convert("RGB")
        plt.close(fig)
        steps.append((f"🔍 Correlation Heatmap #{rank+1}", heat))

        # 3️⃣ ORB matches
        def _prep(i): 
            c = ImageEnhance.Contrast(i).enhance(1.5)
            g = cv2.cvtColor(np.array(c), cv2.COLOR_RGB2GRAY)
            return cv2.GaussianBlur(g, (3,3), 0)
        kp1, d1 = cv2.ORB_create(500).detectAndCompute(_prep(query_img), None)
        kp2, d2 = cv2.ORB_create(500).detectAndCompute(_prep(match_img), None)
        if d1 is not None and d2 is not None:
            matches = sorted(
                cv2.BFMatcher(cv2.NORM_HAMMING, True).match(d1, d2),
                key=lambda m: m.distance
            )[:25]
            orb_img = cv2.drawMatches(
                np.array(query_img), kp1,
                np.array(match_img), kp2,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            steps.append((f"🔗 ORB Matches #{rank+1}", Image.fromarray(orb_img)))

        # 4️⃣ Initial overlay + dynamic red line
        w, h = query_img.size
        split = _find_split_column(query_img)
        overlay = query_img.convert("RGBA")
        fc = match_img.resize((w,h)).crop((split,0,w,h)).convert("RGBA")
        mask = fc.split()[-1].point(lambda p: 128)
        overlay.paste(fc, (split,0), mask)
        d = ImageDraw.Draw(overlay)
        d.line([(split,0),(split,h)], fill=(255,0,0), width=2)
        ov_rgb = overlay.convert("RGB")
        steps.append((f"📈 Overlay #{rank+1}", ov_rgb))

        # 5️⃣ For the best match, trace & crop
        if rank == 0:
            # left/right regions
            left  = ov_rgb.crop((0,0,split,h))
            right = ov_rgb.crop((split,0,w,h))
            ar, (rh, rw) = np.array(right), right.size[::-1]
            y0, y1 = int(0.1*rh), int(0.9*rh)
            gray = np.dot(ar[y0:y1], [0.299,0.587,0.114])
            dark = gray < 200
            mp = int(0.01*(y1-y0))
            coords = [
                (x, int(np.median(np.where(dark[:,x])[0]))+y0)
                for x in range(rw)
                if len(np.where(dark[:,x])[0]) >= mp
            ]
            al = np.array(left)
            grayl = np.dot(al[y0:y1], [0.299,0.587,0.114])
            darkl = grayl < 200
            ys = np.where(darkl[:, -2])[0]
            ymed = (int(np.median(ys))+y0) if ys.size >= mp else (y0+y1)//2
            full = [(0, ymed)] + coords

            blank = Image.new("RGB", (rw, rh), (255,255,255))
            d2 = ImageDraw.Draw(blank)
            if full:
                d2.line(full, fill=(0,0,255), width=3)
            steps.append(("🔵 Forecast Line Only", blank))

            combo = Image.new("RGB", (w, h))
            combo.paste(left, (0,0))
            combo.paste(blank, (split,0))
            cx, cy = int(0.1*w), int(0.1*h)
            final = combo.crop((cx, cy, w-cx, h-cy))
            steps.append(("Final Cropped Output", final))
            final_overlay = final

    return (final_overlay, steps) if debug else final_overlay
