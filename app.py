# ✅ Debug Mode ON + Forecast Wrapped in Try/Except

from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import os
from werkzeug.utils import secure_filename
from forecast_engine import process_forecast_pipeline

app = Flask(__name__)
app.config['DEBUG'] = True  # 🔍 Enable debug mode for error tracebacks

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "query_image" not in request.files:
            return "No file uploaded", 400
        file = request.files["query_image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        query_img = Image.open(filepath).convert("RGB")

        try:
            result_img = process_forecast_pipeline(query_img)
        except Exception as e:
            print("🔥 Forecast pipeline crashed:", str(e))
            raise

        result_path = os.path.join(STATIC_FOLDER, "result.png")
        result_img.save(result_path)

        return render_template("index.html", result_img="result.png")

    return render_template("index.html", result_img=None)

@app.route("/reload", methods=["POST"])
def reload_model():
    return redirect(url_for("index"))

# Run via werkzeug to force correct port binding on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", port, app)
