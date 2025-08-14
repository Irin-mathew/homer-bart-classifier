from flask import Flask, render_template, request
from fastai.learner import load_learner
from pathlib import Path, PosixPath, WindowsPath
import os

# Patch PosixPath to WindowsPath if necessary
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_path = Path("model.pkl")
learn = load_learner(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Make prediction
        pred_class, pred_idx, probs = learn.predict(filepath)
        return render_template('index.html', filename=file.filename, prediction=pred_class)

@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="300">'

if __name__ == '__main__':
    app.run(debug=True)
