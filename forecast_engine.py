import os
import torch
from PIL import Image
from utils import extract_embedding, find_best_match, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    # 1) Load embeddings
    embeddings_path = os.path.join("data", "embeddings.pth")
    data_embeddings = torch.load(embeddings_path)
    print(f"🔢 Loaded {len(data_embeddings)} embeddings")

    # 2) Extract embedding from the query image
    query_embedding = extract_embedding(query_img)
    print("🎯 Query embedding computed")

    # 3) Find best match manually to log score
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb).item()
        if score < best_score:
            best_score = score
            best_match = fname
    print(f"🤝 Best match: {best_match} with score {best_score:.4f}")

    # 4) Construct match path
    match_path = os.path.join("data", "screenshots", best_match)
    if not os.path.isfile(match_path):
        raise FileNotFoundError(f"Expected screenshot not found on disk: {match_path}")

    # 5) Open matched image and generate overlay forecast
    match_img = Image.open(match_path).convert("RGB")
    result_img = generate_overlay_forecast(query_img, match_img)
    return result_img
