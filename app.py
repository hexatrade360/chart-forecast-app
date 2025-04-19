import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from forecast_engine import process_forecast_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Check if file is part of the request
            if "query" not in request.files:
                print("❌ No file part in the request.")
                return "No file part in request", 400

            file = request.files["query"]

            # Check if user submitted an empty file
            if file.filename == "":
                print("❌ No file selected for upload.")
                return "No file selected", 400

            # Load image
            query_img = Image.open(file.stream).convert("RGB")
            print("🟡 POST request received and image loaded.")

            # Run forecast pipeline
            overlay, steps = process_forecast_pipeline(query_img, debug=True)
            print("✅ Forecast pipeline completed successfully.")

            # Save the final result for display
            result_path = os.path.join("static", "result.png")
            overlay.save(result_path)
            print(f"💾 Saved final result to {result_path}")

            # Prepare HTML with steps and result
            return render_template("result.html", result_img="result.png", steps=steps)

        except Exception as e:
            print(f"🔥 Forecast pipeline crashed: {e}")
            return "Internal Server Error", 500

    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True)
