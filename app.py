
import os
from flask import Flask, request, render_template
from forecast_engine import process_forecast_pipeline
from PIL import Image
import uuid

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    debug_steps = []
    debug = False

    if request.method == "POST":
        print("🟡 POST request received")
        if "query_image" not in request.files:
            print("❌ No file part in the request.")
            return "❌ No file part", 400
        file = request.files["query_image"]
        if file.filename == "":
            print("❌ No selected file.")
            return "❌ No selected file", 400
        if file:
            filename = file.filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            print(f"📂 Saved uploaded file to: {path}")
            img = Image.open(path).convert("RGB")
            debug = "debug" in request.form

            try:
                result = process_forecast_pipeline(img, debug=debug)
                if debug:
                    result_img, steps = result
                    uid = str(uuid.uuid4().hex)
                    debug_steps = []
                    for i, (label, step_img) in enumerate(steps):
                        step_path = f"static/step_{uid}_{i}.png"
                        step_img.save(step_path)
                        debug_steps.append({"label": label, "url": step_path})
                else:
                    result_img = result
                result_name = f"result_{uuid.uuid4().hex}.png"
                result_path = os.path.join("static", result_name)
                result_img.save(result_path)
                print(f"💾 Saved result to: {result_path}")
                return render_template("index.html", result_img=result_name,
                                       debug=debug, debug_steps=debug_steps)
            except Exception as e:
                print(f"🔥 Forecast error: {e}")
                return f"🔥 Forecast error: {e}", 500

    return render_template("index.html", result_img=None, debug=debug, debug_steps=[])

if __name__ == "__main__":
    app.run(debug=True)
