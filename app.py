# ✅ Version: manual-red-line v1.1 — stable forecast overlay with manual red line support

from flask import Flask, render_template, request, send_file
from PIL import Image
from forecast_engine import process_forecast_pipeline
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query"]
        if file:
            img = Image.open(file.stream).convert("RGB")
            result_img = process_forecast_pipeline(img)
            result_path = "static/result.png"
            result_img.save(result_path)
            return render_template("index.html", result_img=result_path)
    return render_template("index.html", result_img=None)
