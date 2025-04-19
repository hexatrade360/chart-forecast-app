import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, generate_overlay_forecast

def process_forecast_pipeline(query_img: Image.Image) -> Image.Image:
    # Load embeddings
    emb_path = os.path.join("data", "embeddings.pth")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"embeddings.pth not found: {emb_path}")
    data_embeddings = torch.load(emb_path)

    # Prepare arrays
    names = list(data_embeddings.keys())
    vecs  = np.stack([data_embeddings[n].numpy() for n in names])

    # Embed query and find best match
    q = extract_embedding(query_img).numpy()[None, :]
    sims = cosine_similarity(q, vecs)[0]
    idx  = int(np.argmax(sims))
    best = names[idx]
    print(f"🤝 Best match: {best} (cosine={sims[idx]:.4f})")

    # Load match image and overlay forecast
    match_img = Image.open(os.path.join("data","screenshots", best)).convert("RGB")
    return generate_overlay_forecast(query_img, match_img)
