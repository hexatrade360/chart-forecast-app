import os
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from forecast_engine import process_forecast_pipeline

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'screenshot' not in request.files:
            return '❌ No file part'
        file = request.files['screenshot']
        if file.filename == '':
            return '❌ No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = Image.open(file_path).convert("RGB")
                overlay, steps = process_forecast_pipeline(img, debug=True)

                # Save result image
                output_path = os.path.join(app.config['STATIC_FOLDER'], f"result_{filename}")
                overlay.save(output_path)

                # Build HTML debug output
                html = f'<h2>✅ Forecast generated for: {filename}</h2>'
                html += f'<h3>📈 Final Forecast Result:</h3><img src="/static/result_{filename}" width="600"><hr>'

                for title, step_img in steps:
                    step_name = f"{title.replace(' ','_').replace('#','')}_{filename}"
                    step_path = os.path.join(app.config['STATIC_FOLDER'], step_name + ".png")
                    step_img.save(step_path)
                    html += f'<h4>{title}</h4><img src="/static/{step_name}.png" width="600"><br><br>'

                return render_template_string(html)
            except Exception as e:
                return f'🔥 Forecast pipeline crashed: {str(e)}'
    return '''
    <!doctype html>
    <title>Chart Forecast App</title>
    <h1>📤 Upload a 24-Hour Chart Screenshot</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=screenshot>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
