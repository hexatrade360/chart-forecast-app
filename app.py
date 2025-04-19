# ✅ app.py with cache-busting unique result filenames

from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import os
import uuid
from werkzeug.utils import secure_filename
from forecast_engine import process_forecast_pipeline

app = Flask(__name__)
app.config['DEBUG'] = True

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    if request.method == "POST":
        print("🟡 POST request received")
        if "query_image" not in request.files:
            print("❌ No file in request")
            return "No file uploaded", 400

        file = request.files["query_image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"📂 Saved uploaded file to: {filepath}")

        try:
            query_img = Image.open(filepath).convert("RGB")
            print("🧠 Loaded image into PIL")

            result_img_obj = process_forecast_pipeline(query_img)
            print("✅ Forecast pipeline returned result")

            # use a unique filename to avoid caching issues
            result_filename = f"result_{uuid.uuid4().hex}.png"
            result_path = os.path.join(STATIC_FOLDER, result_filename)
            result_img_obj.save(result_path)
            print("💾 Saved result to:", result_path)

            result_img = result_filename

        except Exception as e:
            print("🔥 Forecast pipeline crashed:", str(e))
            raise

    return render_template("index.html", result_img=result_img)

@app.route("/reload", methods=["POST"])
def reload_model():
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", port, app)
