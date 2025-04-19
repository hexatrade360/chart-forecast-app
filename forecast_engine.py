import os
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    # Load embeddings
    emb_path = os.path.join("data", "embeddings.pkl")
    with open(emb_path, "rb") as f:
        data_embeddings = pickle.load(f)
    filenames = list(data_embeddings.keys())
    vectors = np.stack([data_embeddings[f] for f in filenames])

    # Compute query embedding
    q = extract_embedding(query_img).numpy()[None, :]

    # Find best match
    sims = cosine_similarity(q, vectors)[0]
    best_idx = int(np.argmax(sims))
    best_name, best_score = filenames[best_idx], sims[best_idx]
    print(f"🤝 Best match: {best_name} (cosine={best_score:.4f})")

    # Load best image
    match_img = Image.open(os.path.join("data","screenshots",best_name)).convert("RGB")

    # Overlay forecast
    return generate_overlay_forecast(query_img, match_img)
