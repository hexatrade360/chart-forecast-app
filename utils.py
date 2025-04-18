import numpy as np
from PIL import Image
import cv2
import torch
import os

def extract_embedding(img):
    # Convert to grayscale and resize to consistent dimensions
    img = img.convert("L").resize((128, 128))
    arr = np.array(img) / 255.0
    return torch.tensor(arr.flatten(), dtype=torch.float)

def find_best_match(query_embedding, data_embeddings):
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb)
        if score < best_score:
            best_score = score
            best_match = fname
    return os.path.join("data", "screenshots", best_match)

def generate_overlay_forecast(query_img, match_img):
    query_np = np.array(query_img.resize((800, 400)))
    match_np = np.array(match_img.resize((800, 400)))
    combined = np.vstack((query_np[:200], match_np[200:]))
    return Image.fromarray(combined)
