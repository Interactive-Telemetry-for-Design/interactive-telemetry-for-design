import re
import random
from flask import Flask, abort, render_template, jsonify, request, send_from_directory
import os
from dotenv import load_dotenv
from config import config

UPLOADS_DIR = config.DATA_DIR / 'uploads'

load_dotenv('.env')

app = Flask(__name__)
app.config['ENV'] = os.getenv('FLASK_ENV')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG') == '1'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['FLASK_RUN_PORT'] = os.getenv('FLASK_RUN_PORT')

@app.route('/')
def index():
    return "<p>Index Page</p>"

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET':
        return render_template('training.html')
    elif request.method == 'POST':
        return process_blocks()
    else:
        abort(405)

@app.route('/process_blocks', methods=['POST'])
def process_blocks():
    """
    Returns a single 'predictions' list of dicts for AI and Ci:
    Each dict => {
      'frame_number': int,
      'label': str,
      'confidence': float,
      'source': 'AI'|'Ci'
    }

    We define two intervals for both AI and Ci:
      frames 1..600 => label=Label1
      frames 601..1000 => label=Label2
    """
    data = request.json
    blocks = data.get("blocks", [])
    epochs = data.get("epochs", 5)

    print('Received blocks from GT:', blocks)
    print('Requested epochs:', epochs)

    # same label intervals for AI vs Ci => chunk boundaries match
    predictions = []

    # AI data
    for f in range(1, 601):
        conf = random.uniform(0.4, 0.8)
        predictions.append({
            "frame_number": f,
            "label": "Label1",
            "confidence": conf,
            "source": "AI"
        })
    for f in range(601, 1001):
        conf = random.uniform(0.2, 0.6)
        predictions.append({
            "frame_number": f,
            "label": "Label2",
            "confidence": conf,
            "source": "AI"
        })

    # Ci data
    for f in range(1, 601):
        conf = random.uniform(0.7, 1.0)
        predictions.append({
            "frame_number": f,
            "label": "Label1",
            "confidence": conf,
            "source": "Ci"
        })
    for f in range(601, 1001):
        conf = random.uniform(0.3, 0.7)
        predictions.append({
            "frame_number": f,
            "label": "Label2",
            "confidence": conf,
            "source": "Ci"
        })

    return jsonify({
        "status": "success",
        "predictions": predictions
    })

@app.route('/finished.html')
def finished_page():
    return "<h1>Finished Page</h1><p>You have reached the finished page.</p>"

@app.route('/uploads/<path:filename>')
def serve_video(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        abort(404, description='Video file not found.')
    return send_from_directory(UPLOADS_DIR, filename)

if __name__ == '__main__':
    app.run()
