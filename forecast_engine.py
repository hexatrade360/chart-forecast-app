import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    data_embeddings = torch.load(os.path.join("data","embeddings.pth"))
    fnames = list(data_embeddings.keys())
    vecs = np.stack([data_embeddings[f].numpy() for f in fnames])
    q = extract_embedding(query_img).numpy()[None,:]
    sims = cosine_similarity(q,vecs)[0]
    idx = int(np.argmax(sims))
    print(f"🤝 Best match: {fnames[idx]} (cosine={sims[idx]:.4f})")
    match = Image.open(os.path.join("data","screenshots",fnames[idx])).convert("RGB")
    return generate_overlay_forecast(query_img, match)
