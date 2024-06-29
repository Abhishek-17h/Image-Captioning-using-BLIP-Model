from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import os
import numpy as np
from model_setup import generate_caption

app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        # Generate caption for the uploaded image
        caption = generate_caption(image_path)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
