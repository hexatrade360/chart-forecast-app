import os, uuid
from flask import Flask, request, render_template
from PIL import Image
from forecast_engine import process_forecast_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'query' not in request.files:
            print("❌ No file part in the request.")
            return "❌ No file part in the request.", 400

        file = request.files['query']
        if file.filename == '':
            print("❌ No selected file.")
            return "❌ No selected file.", 400

        filename = os.path.basename(file.filename)
        save_path = os.path.join("uploads", filename)
        file.save(save_path)
        print(f"📂 Saved uploaded file to: {save_path}")

        img = Image.open(save_path).convert("RGB")
        debug_mode = 'debug' in request.form

        try:
            result_img, steps = process_forecast_pipeline(img, debug=debug_mode)
        except Exception as e:
            print(f"🔥 Forecast pipeline crashed: {e}")
            return f"🔥 Forecast error: {e}", 500

        result_name = f"result_{uuid.uuid4().hex}.png"
        result_path = os.path.join("static", result_name)
        result_img.save(result_path)
        print(f"💾 Saved result to: {result_path}")

        if debug_mode:
            debug_steps = []
            for label, step_img in steps:
                debug_name = f"step_{uuid.uuid4().hex}.png"
                step_path = os.path.join("static", debug_name)
                step_img.save(step_path)
                debug_steps.append({"label": label, "url": f"static/{debug_name}"})
            return render_template("index.html", debug=True, debug_steps=debug_steps)

        return render_template("index.html", debug=False, result_img=result_name)
    return render_template("index.html")