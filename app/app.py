import re
from flask import Flask, Response, abort, render_template, jsonify, request, send_file, send_from_directory
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
    return render_template('index.html')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    match request.method:
        case 'GET':
            return render_template('training.html')
        case 'POST':
            return process_blocks()
        case _:
            abort(405)

@app.route('/process_blocks', methods=['POST'])
def process_blocks():
    blocks = request.json
    # Process the blocks as needed
    print('Received blocks:', blocks)
    return jsonify({'status': 'success'})
    

@app.route('/uploads/<path:filename>')
def serve_video(filename: str):
    # TODO: No check that the requested filename is a video file
    file_path = UPLOADS_DIR / filename

    if not file_path.exists():
        abort(404, description='Video file not found.')

    return send_from_directory(UPLOADS_DIR, filename)
    # return send_file(config.DATA_DIR / filename, mimetype='video/mp4') 

if __name__ == '__main__':
    app.run()
