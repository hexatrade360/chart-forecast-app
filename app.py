from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import os
import uuid

from forecast_engine import process_forecast_pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    debug = False
    debug_steps = []

    if request.method == "POST":
        # Check debug mode checkbox
        debug = request.form.get("debug") == "on"

        # Save uploaded file
        file = request.files.get("query_image")
        if not file:
            return "No file uploaded", 400
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Open and process
        query_img = Image.open(input_path).convert("RGB")
        if debug:
            # Returns overlay + list of (label, PIL.Image)
            overlay, steps = process_forecast_pipeline(query_img, debug=True)
            # Save each debug step
            uid = uuid.uuid4().hex
            for i, (label, img) in enumerate(steps):
                outname = f"debug_{uid}_{i}.png"
                outpath = os.path.join(STATIC_FOLDER, outname)
                img.save(outpath)
                debug_steps.append({
                    "label": label,
                    "url": f"static/{outname}"
                })
        else:
            # Just the final overlay
            overlay = process_forecast_pipeline(query_img, debug=False)

        # Save final overlay
        outname = f"result_{uuid.uuid4().hex}.png"
        outpath = os.path.join(STATIC_FOLDER, outname)
        overlay.save(outpath)
        result_img = outname

    return render_template(
        "index.html",
        result_img=result_img,
        debug=debug,
        debug_steps=debug_steps
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
