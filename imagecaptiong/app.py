import os
from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
from inference import caption_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DEFAULT = os.path.join(BASE_DIR, 'models', 'caption.pth')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image Captioning UI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; padding: 1rem; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    input[type=file] { margin-top: 0.5rem; }
    button { padding: 0.6rem 1rem; margin-top: 1rem; }
    img { max-width: 100%; height: auto; margin-top: 1rem; }
    .caption { margin-top: 1rem; font-size: 1.1rem; font-weight: bold; }
    .warning { color: #b00; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Image Caption Generator</h1>
    <p>Upload an image and generate a caption using the pretrained model.</p>
    {% if warning %}
      <div class="warning">{{ warning }}</div>
    {% endif %}
    <form method="post" enctype="multipart/form-data">
      <label>Choose image file:</label><br />
      <input type="file" name="image" accept="image/*" required><br />
      <label>Model path:</label><br />
      <input type="text" name="model_path" value="{{ model_path }}" style="width:100%;" /><br />
      <button type="submit">Generate caption</button>
    </form>
    {% if image_url %}
      <img src="{{ image_url }}" alt="Uploaded image" />
    {% endif %}
    {% if caption %}
      <div class="caption">Caption: {{ caption }}</div>
    {% endif %}
  </div>
</body>
</html>
'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_url = None
    warning = None
    model_path = MODEL_DEFAULT

    if request.method == 'POST':
        if 'image' not in request.files:
            warning = 'No image file part in the request.'
        else:
            file = request.files['image']
            model_path = request.form.get('model_path', MODEL_DEFAULT) or MODEL_DEFAULT
            if file.filename == '':
                warning = 'No image selected.'
            elif not allowed_file(file.filename):
                warning = 'Only PNG, JPG, JPEG and GIF files are allowed.'
            else:
                filename = os.path.basename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_url = url_for('uploaded_file', filename=filename)
                if not os.path.exists(model_path):
                    warning = (
                        f'Model not found at {model_path}.\n'
                        'Please train the model first or provide a valid .pth path.\n'
                        'Example: python image_captiong.py train --data-dir data --captions captions.txt --epochs 10 --save-path imagecaptiong/models/caption.pth'
                    )
                else:
                    try:
                        caption = caption_image(save_path, model_path)
                    except Exception as exc:
                        warning = f'Error generating caption: {exc}'

    return render_template_string(TEMPLATE, caption=caption, image_url=image_url, warning=warning, model_path=model_path)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/infer', methods=['GET'])
def infer():
    image_path = request.args.get('image')
    model_path = request.args.get('model', MODEL_DEFAULT)
    if not image_path:
        return 'No image provided.', 400
    if not model_path:
        return 'No model provided.', 400
    try:
        caption = caption_image(image_path, model_path)
    except Exception as exc:
        return f'Error generating caption: {exc}', 500
    return caption


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
