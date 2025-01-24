import random
from flask import Flask, jsonify, request, render_template, redirect, send_from_directory, url_for, flash, abort
from src.plotting import prepare_data
from src.imu_extraction import extract_imu_data
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from config import config
from dotenv import load_dotenv
from src import labeler
from src import model as modelfunc
from src.model import save_model, load_model
from pprint import pprint

load_dotenv('.env')

app = Flask(__name__)
app.config['ENV'] = os.getenv('FLASK_ENV')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG') == '1'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['FLASK_RUN_PORT'] = os.getenv('FLASK_RUN_PORT')

app.config['UPLOAD_FOLDER'] = config.DATA_DIR / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1024 * 1024 # 50gb limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

df = None
settings = {}
stored_sequences = None
label_mapping = pd.DataFrame()
model = None

@app.route('/')
def upload_page():
    return render_template('welcome_tab.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    global df
    df = labeler.frame_index(settings['video_path'], df)
    if request.method == 'GET':
        print(settings)
        return render_template('training.html', video_src=f'/uploads/{Path(settings['video_path']).name}')
    elif request.method == 'POST':
        return process_blocks()
    else:
        abort(405)

@app.route('/predict_continue.html', methods=['GET'])
def predict_continue():
    return render_template('predict_continue.html', show_model_upload=True)

@app.route('/alternative_continue', methods=['GET'])
def predict_cont_no_show():
    return render_template('predict_continue.html', show_model_upload=False)


@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html', video_src=f'/uploads/cut-GH010039-0.2.mp4')
    # TODO:
    return render_template('predict.html', video_src=f'/uploads/{Path(file_paths['mp4_path']).name}')

@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        save_model(model, label_mapping, settings, stored_sequences)
        flash('Model saved succesfuly as model.zip')
    except Exception as e:
        flash('Could not save model')
        print(f'model save failed: {e}')
    return render_template('predict_continue.html', show_model_upload=False)

@app.route('/process_blocks', methods=['POST'])
def process_blocks():
    global stored_sequences
    global label_mapping
    global df
    global settings
    global prediction_df
    global model
    global padded_sequences
    global padded_labels


    data = request.json
    blocks = data.get("blocks", [])
    # blocks = [{'label': 'walking', 'frame_start': 0, 'frame_end': 2640}, {'label': 'left', 'frame_start': 2820, 'frame_end': 3255}, {'label': 'rechtdoor', 'frame_start': 3255, 'frame_end': 4830}, {'label': 'right', 'frame_start': 4830, 'frame_end': 5160}, {'label': 'rechtdoor', 'frame_start': 8099, 'frame_end': 12735}, {'label': 'stopping', 'frame_start': 12735, 'frame_end': 13110}, {'label': 'standing', 'frame_start': 13110, 'frame_end': 13800}, {'label': 'right', 'frame_start': 18780, 'frame_end': 19035}]
    epochs = data.get("epochs", 5)
    settings['epochs'] = epochs
    pprint(df.head(10)) # test
    results = modelfunc.run_model(blocks, settings, model=model, unlabeled_df=df, label_mapping=label_mapping, stored_sequences=stored_sequences)
    result_list, settings, model, prediction_df, label_mapping, df, padded_sequences, padded_labels = results
    print(len(result_list))

    print('Received blocks from GT:', blocks)
    print('Requested epochs:', epochs)

    # same label intervals for AI vs Ci => chunk boundaries match


    predictions = []

    for result in result_list:
        predictions.append({
            "frame_number": result['frame_number'],
            "label": result['label'],
            "confidence": result['confidence'],
            "source": 'AI',
        })
        predictions.append({
            "frame_number": result['frame_number'],
            "label": result['label'],
            "confidence": result['confidence'],
            "source": 'Ci',
        })

    # # AI data
    # for f in range(1, 601):
    #     conf = random.uniform(0.4, 0.8)
    #     predictions.append({
    #         "frame_number": f,
    #         "label": "Label1",
    #         "confidence": conf,
    #         "source": "AI"
    #     })
    # for f in range(601, 1001):
    #     conf = random.uniform(0.2, 0.6)
    #     predictions.append({
    #         "frame_number": f,
    #         "label": "Label2",
    #         "confidence": conf,
    #         "source": "AI"
    #     })

    # # Ci data
    # for f in range(1, 601):
    #     conf = random.uniform(0.7, 1.0)
    #     predictions.append({
    #         "frame_number": f,
    #         "label": "Label1",
    #         "confidence": conf,
    #         "source": "Ci"
    #     })
    # for f in range(601, 1001):
    #     conf = random.uniform(0.3, 0.7)
    #     predictions.append({
    #         "frame_number": f,
    #         "label": "Label2",
    #         "confidence": conf,
    #         "source": "Ci"
    #     })

    return jsonify({
        "status": "success",
        "predictions": predictions
    })

@app.route('/data/uploads', methods=['POST'])
def handle_upload():
    global df
    global settings

    # Handle file uploads
    mp4_file = request.files.get('mp4-upload')
    csv_file = request.files.get('CSV-upload')

    # Get form data
    settings = {
        'overlap': float(request.form.get('overlap', 0.50)),
        'length': int(request.form.get('length', 10)),
        'batch_size': int(request.form.get('batch_size', 8)),
        'dropout': float(request.form.get('dropout', 0.2)),
        'LSTM_units': int(request.form.get('LSTM_units', 256)),
        'learning_rate': float(request.form.get('learning_rate', 0.0001)),
        "epochs": 5,
        "dense_activation": "softmax",
        "LSTM_activation": "tanh",
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "from_scratch": True,
        'target_sequence_length': None,
        "weight_decay": None
    }
    
    # Save files and get their paths
    if mp4_file and mp4_file.filename:
        app.config['UPLOAD_FOLDER']
        mp4_path = app.config['UPLOAD_FOLDER'] / mp4_file.filename
        mp4_file.save(mp4_path)
        settings['video_path'] = mp4_path
        
        
    if csv_file and csv_file.filename:
        if not mp4_file:
            flash('Error: MP4 file is required when uploading a CSV.')
            return redirect(url_for('upload_page'))
        csv_path = app.config['UPLOAD_FOLDER'] / csv_file.filename
        csv_file.save(csv_path)
        settings['imu_path'] = csv_path

    if 'video_path' in settings and 'imu_path' in settings:
        try:
            df_t = pd.read_csv(settings['imu_path'])
        except Exception as e:
            print(f"Error while reading CSV: {e}")
            flash("An error occurred while reading the CSV file.")
            return redirect(url_for('upload_page'))
    
        required_columns = ["TIMESTAMP", "ACCL_x", "ACCL_y", "ACCL_z", "GYRO_x", "GYRO_y", "GYRO_z"]

        # Create a new DataFrame with only the required columns in the specified order
        df_t2 = pd.DataFrame({col: df_t[col] for col in required_columns if col in df_t.columns})

        # Check if any required columns are missing
        if list(df_t2.columns) != required_columns:
            flash("CSV file does not have the required columns (Capital sensitive): TIMESTAMP, ACCL_x, ACCL_y, ACCL_z, GYRO_x, GYRO_y, GYRO_z", "error")
            return redirect(url_for('upload_page'))
        df = df_t2
        return redirect(url_for('training'))

    # If 'mp4_path' is present, run the extract method
    if 'video_path' in settings and not 'imu_path' in settings:
        try:
            df = extract_imu_data(settings['video_path'])
            return redirect(url_for('training'))
        except Exception as e:
            print(f"Error while reading extracting IMU data: {e}")
            flash("An error occurred while trying to extract the IMU data.")
            return redirect(url_for('upload_page'))
        
    flash("You need to upload something")
    return redirect(url_for('upload_page'))


@app.route('/upload_files', methods=['POST'])
def upload_files():
    global df
    global settings
    model_file = request.files.get('model_upload')
    video_file = request.files.get('mp4_upload')
    csv_file = request.files.get('csv_upload')
    should_reroute = False  # Used to decide to which page to reroute for showing model upload or not
    if model_file:
        should_reroute = True
    action = request.form.get('action')

    if model_file and model_file.filename:
        global model
        global label_mapping
        global settings
        global stored_sequences

        app.config['UPLOAD_FOLDER']
        model_path = app.config['UPLOAD_FOLDER'] / model_file.filename
        model_file.save(model_path)
        settings['model_path'] = str(model_path)
        try:
            model, label_mapping, settings, stored_sequences = load_model(str(model_path))
        except Exception as e:
            flash(Error with uploaded zip)
            print(e)
            return redirect(url_for('predict_continue'))

    
    if video_file and video_file.filename:
        app.config['UPLOAD_FOLDER']
        mp4_path = app.config['UPLOAD_FOLDER'] / video_file.filename
        video_file.save(mp4_path)
        settings['video_path'] = str(mp4_path)
        
    if csv_file and csv_file.filename:
        if not video_file:
            flash('Error: MP4 file is required when uploading a CSV.')
            if should_reroute:
                return redirect(url_for('predict_continue'))
            return redirect(url_for('predict_cont_no_show'))
        csv_path = app.config['UPLOAD_FOLDER'] / csv_file.filename
        csv_file.save(csv_path)
        settings['imu_path'] = str(csv_path)

    if 'video_path' in settings and 'imu_path' in settings:
        try:
            df_t = pd.read_csv(settings['imu_path'])
        except Exception as e:
            print(f"Error while reading CSV: {e}")
            flash("An error occurred while reading the CSV file.")
            if should_reroute:
                return redirect(url_for('predict_continue'))
            return redirect(url_for('predict_cont_no_show'))
    
        required_columns = ["TIMESTAMP", "ACCL_x", "ACCL_y", "ACCL_z", "GYRO_x", "GYRO_y", "GYRO_z"]

        # Create a new DataFrame with only the required columns in the specified order
        df_t2 = pd.DataFrame({col: df_t[col] for col in required_columns if col in df_t.columns})

        # Check if any required columns are missing
        if list(df_t2.columns) != required_columns:
            flash("CSV file does not have the required columns (Capital sensitive): TIMESTAMP, ACCL_x, ACCL_y, ACCL_z, GYRO_x, GYRO_y, GYRO_z", "error")
            if should_reroute:
                return redirect(url_for('predict_continue'))
            return redirect(url_for('predict_cont_no_show'))
        df = df_t2

    # If 'mp4_path' is present, run the extract method
    if 'video_path' in settings and not 'imu_path' in settings:
        try:
            df = extract_imu_data(settings['video_path'])
        except Exception as e:
            print(f"Error while reading extracting IMU data: {e}")
            flash("An error occurred while trying to extract the IMU data.")
            if should_reroute:
                return redirect(url_for('predict_continue'))
            return redirect(url_for('predict_cont_no_show'))

    if action == 'predict':
        return redirect(url_for('predict'))
    elif action == 'continue_training':
        return redirect(url_for('training'))
    
    if should_reroute:
        return redirect(url_for('predict_continue'))
    return redirect(url_for('predict_cont_no_show'))

@app.route('/plot')
def plot():
    return render_template('plot.html')

@app.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    try:
        data = request.json
        print("Received data:", data)  # Debugging: Check payload
        x_col = data.get('x')
        y_col = data.get('y')
        ci = float(data.get('ci', 0))  # Ensure 'ci' is a float
    except Exception as e:
        print("Error parsing request data:", str(e))
        return {"error": "Invalid input data"}, 400
    print(ci)
    if 'CONFIDENCE' in df.columns:
        principal_df, mapping = prepare_data(df.copy(), ci)
    else:
        df['CONFIDENCE'] = np.random.rand(len(df))
        principal_df, mapping = prepare_data(df.copy(),ci)

    
    


    data = {
        'x': principal_df[x_col].tolist(),
        'y': principal_df[y_col].tolist(),
        'frames': principal_df['TIMESTAMP'].tolist(),
        'colours': principal_df['COLOUR'].tolist(),
        'legend': mapping
    }
    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

@app.route('/uploads/<path:filename>')
def serve_video(filename: str):
    file_path = app.config['UPLOAD_FOLDER'] / filename
    if not file_path.exists():
        abort(404, description='Video file not found.')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)