import os
import torch
from PIL import Image
from utils import extract_embedding, find_best_match, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    # 1) Ensure embeddings file exists
    embedding_path = os.path.join("data", "embeddings.pth")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError("embeddings.pth not found in /data")

    # 2) Load precomputed embeddings
    data_embeddings = torch.load(embedding_path)

    # 3) Extract embedding from the query image
    query_embedding = extract_embedding(query_img)

    # 4) Find the best matching historical image path
    #    (find_best_match should return the full path under data/screenshots/)
    match_path = find_best_match(query_embedding, data_embeddings)

    # 5) Safety check: ensure the matched file exists
    if not os.path.isfile(match_path):
        raise FileNotFoundError(f"Matched image not found on disk: {match_path}")

    # 6) Open and convert the matched image
    match_img = Image.open(match_path).convert("RGB")

    # 7) Generate and return the overlay forecast
    return generate_overlay_forecast(query_img, match_img)
