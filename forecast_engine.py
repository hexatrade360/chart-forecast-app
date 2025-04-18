import os
import torch
from PIL import Image
from utils import extract_embedding, find_best_match, generate_overlay_forecast

def process_forecast_pipeline(query_img):
    embedding_path = os.path.join("data", "embeddings.pth")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError("embeddings.pth not found in /data")

    # Load precomputed embeddings
    data_embeddings = torch.load(embedding_path)

    # Extract embedding from the query image
    query_embedding = extract_embedding(query_img)

    # Find the best matching historical image
    match_path = find_best_match(query_embedding, data_embeddings)
    match_img = Image.open(match_path).convert("RGB")

    # Generate the overlay image as forecast
    result_img = generate_overlay_forecast(query_img, match_img)
    return result_img
