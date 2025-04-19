
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding

def process_forecast_pipeline(query_img, top_k=3, debug=False):
    emb_path = os.path.join("data","embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs = np.stack([data_embeddings[n].detach().numpy() for n in names])

    q = extract_embedding(query_img).detach().numpy()[None,:]
    sims = cosine_similarity(q, vecs)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]

    steps = []
    final_overlay = None

    for rank, idx in enumerate(top_idxs):
        name = names[idx]
        match_img = Image.open(os.path.join("data","screenshots",name)).convert("RGB")

        # Initial overlay with no extra red line drawn
        w,h = query_img.size
        mid = w//2
        overlay = query_img.convert("RGBA")
        forecast_crop = match_img.resize((w,h)).crop((mid,0,w,h)).convert("RGBA")
        mask = forecast_crop.split()[-1].point(lambda p:128)
        overlay.paste(forecast_crop, (mid,0), mask)

        ov_rgb = overlay.convert("RGB")
        if rank == 0:
            # Blue forecast line logic
            arr = np.array(ov_rgb)
            rm = (arr[:,:,0]>200)&(arr[:,:,1]<80)&(arr[:,:,2]<80)
            prop = rm.sum(axis=0)/h
            cols = np.where(prop>0.02)[0]
            split = cols[-1] if len(cols)>=2 else int(np.argmax(prop))
            left = ov_rgb.crop((0,0,split+1,h))
            right = ov_rgb.crop((split+1,0,w,h))
            rh, rw = right.size[1], right.size[0]
            arr_r = np.array(right)
            y0,y1 = int(0.1*rh), int(0.9*rh)
            gray = np.dot(arr_r[y0:y1], [0.299,0.587,0.114])
            dark = gray<200
            minp = int(0.01*(y1-y0))
            coords = [(x, int(np.median(np.where(dark[:,x])[0]))+y0)
                      for x in range(rw) if len(np.where(dark[:,x])[0])>=minp]
            arr_l = np.array(left)
            grayl = np.dot(arr_l[y0:y1], [0.299,0.587,0.114])
            darkl = grayl<200
            ys = np.where(darkl[:,-2])[0]
            y_med = int(np.median(ys))+y0 if len(ys)>=minp else int((y0+y1)/2)
            full = [(0,y_med)] + coords
            blank = Image.new("RGB",(rw,rh),(255,255,255))
            draw2 = ImageDraw.Draw(blank)
            if full:
                draw2.line(full, fill=(0,0,255), width=3)
            combo = Image.new("RGB",(w,h))
            combo.paste(left,(0,0))
            combo.paste(blank,(split+1,0))
            cx,cy = int(0.1*w), int(0.1*h)
            final = combo.crop((cx,cy,w-cx,h-cy))
            final_overlay = final

    return final_overlay if not debug else (final_overlay, steps)
