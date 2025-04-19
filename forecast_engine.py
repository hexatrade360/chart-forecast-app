import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    """
    1) Load embeddings (filename -> tensor)
    2) Compute query embedding
    3) Find best match via cosine similarity
    4) Load match image and pass to overlay logic
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

    # 5) Generate overlay forecast (includes blue line & red split)
    return generate_overlay_forecast(query_img, match_img)
