import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_embedding, generate_overlay_forecast

def process_forecast_pipeline(query_img, debug=False):
    emb_path = os.path.join("data","embeddings.pth")
    data_embeddings = torch.load(emb_path)
    names = list(data_embeddings.keys())
    vecs  = np.stack([data_embeddings[n].numpy() for n in names])
    q = extract_embedding(query_img).numpy()[None,:]
    sims = cosine_similarity(q, vecs)[0]
    idx  = int(np.argmax(sims))
    best = names[idx]
    print(f"🤝 Best match: {best}")
    match_img = Image.open(os.path.join("data","screenshots", best)).convert("RGB")
    final = generate_overlay_forecast(query_img, match_img)
    if debug:
        return final, [("Final Overlay", final)]
    return final
