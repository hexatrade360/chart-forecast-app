
import os
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from forecast_engine import process_forecast_pipeline

app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query"]
        if not file:
            return "❌ No file uploaded", 400

        # Save uploaded file
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)
        print(f"📂 Saved uploaded file to: {filepath}")

        query_img = Image.open(filepath).convert("RGB")

        try:
            overlay, steps = process_forecast_pipeline(query_img, debug=True)
            result_path = os.path.join("static", f"result_{os.path.basename(filepath)}")
            overlay.save(result_path)
            print(f"💾 Saved result to: {result_path}")
        except Exception as e:
            print(f"🔥 Forecast pipeline crashed: {e}")
            return "❌ Forecast pipeline error", 500

        return render_template("index.html", result_image=os.path.basename(result_path), steps=steps)

    return render_template("index.html", result_image=None, steps=[])

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
