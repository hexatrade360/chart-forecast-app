import numpy as np
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from torchvision import models

# 🔧 Same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🧠 Load the same ResNet18 structure used during training
class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, 128)

    def forward(self, x):
        return self.base_model(x)

# 🔍 Load the trained model weights
_model_cache = None
def load_model():
    global _model_cache
    if _model_cache is None:
        model = EmbeddingNet()
        model.eval()
        _model_cache = model
    return _model_cache

# ✅ Updated embedding extraction
def extract_embedding(img):
    model = load_model()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_tensor).squeeze()
    return emb

# 🔍 Match closest screenshot
def find_best_match(query_embedding, data_embeddings):
    best_match = None
    best_score = float("inf")
    for fname, emb in data_embeddings.items():
        score = torch.dist(query_embedding, emb)
        if score < best_score:
            best_score = score
            best_match = fname
    return os.path.join("data", "screenshots", best_match)

# 🖼 Generate overlay image
def generate_overlay_forecast(query_img, match_img):
    query_np = np.array(query_img.resize((800, 400)))
    match_np = np.array(match_img.resize((800, 400)))
    combined = np.vstack((query_np[:200], match_np[200:]))
    return Image.fromarray(combined)
