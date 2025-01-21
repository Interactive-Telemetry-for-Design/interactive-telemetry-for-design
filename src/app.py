from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from src.plotting import prepare_data
from src.imu_extraction import prepare_input
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from config import config

app = Flask(__name__, template_folder='../designer-interface')
app.config['UPLOAD_FOLDER'] = config.DATA_DIR / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1000 * 1024 * 1024 # 50gb limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load and prepare data once when starting the server
df = pd.read_csv(Path('C:/projects/interactive-telemetry-for-design/data/CSVs/GoPro_test.csv'))
principal_df, mapping = prepare_data(df)

@app.route('/')
def upload_page():
    return render_template('welcome_tab.html')

@app.route('/data/uploads', methods=['POST'])
def handle_upload():
    if request.method == 'POST':
        # Handle file uploads
        mp4_file = request.files.get('mp4-upload')
        pt_file = request.files.get('model-upload')
        csv_file = request.files.get('CSV-upload')
        
        # Get form data
        advanced_settings = {
            'a1': request.form.get('a1', 3),
            'a2': request.form.get('a2', 3),
            'a3': request.form.get('a3', 3),
            'a4': request.form.get('a4', 3),
            'a5': request.form.get('a5', 3)
        }
        
        # Save files and get their paths
        file_paths = {}
        
        if mp4_file and mp4_file.filename:
            app.config['UPLOAD_FOLDER']
            mp4_path = app.config['UPLOAD_FOLDER'] / mp4_file.filename
            mp4_file.save(mp4_path)
            file_paths['mp4_path'] = str(mp4_path)
            
        if pt_file and pt_file.filename:
            pt_path = app.config['UPLOAD_FOLDER'] / pt_file.filename
            pt_file.save(pt_path)
            file_paths['pt_path'] = str(pt_path)
            
        if csv_file and csv_file.filename:
            if not mp4_file and mp_file.filename:
                # Validate file uploads
                flash('Error: MP4 file is required when uploading a CSV.')
                return redirect(url_for('welcome_tab.html'))
            csv_path = app.config['UPLOAD_FOLDER'] / csv_file.filename
            csv_file.save(csv_path)
            file_paths['csv_path'] = str(csv_path)

        if 'pt_path' in paths:
            return redirect(url_for('plot'))  # redirect to predict & 
    
        # if 'mp4_path' in paths and 'csv_path' in paths:
        #     try:
        #         df = pd.read_csv(paths['csv_path'])
        #         required_columns = ["TIMESTAMP", "ACCL_x", "ACCL_y", "ACCL_z", "GYRO_x", "GYRO_y", "GYRO_z"]

        #         if list(df.columns) != required_columns:
        #             flash("CSV file does not have the required columns in the correct order: TIMESTAMP, ACCL_x, ACCL_y, ACCL_z, GYRO_x, GYRO_y, GYRO_z")
        #             return redirect(url_for('welcome_tab.html'))
        #         return df
        #     except Exception as e:
        #         print(f"Error while reading CSV: {e}")
        #         flash("An error occurred while reading the CSV file.")
        #         return redirect(url_for('welcome_tab.html'))
            
        # # If 'mp4_path' is present, run the extract method
        # if 'mp4_path' in paths:
        #     df = extract_imu_data(paths['mp4_path'])
        #     return df
        # return redirect(url_for('welcome_tab.html'))


        # Redirect to the plot page
        return redirect(url_for('plot'))

@app.route('/plot')
def plot():
    return Path('designer-interface/plot.html').read_text()

@app.route('/get_plot_data')
def get_plot_data():
    x_col = request.args.get('x', 'PC_1')
    y_col = request.args.get('y', 'PC_2')

    data = {
        'x': principal_df[x_col].tolist(),
        'y': principal_df[y_col].tolist(),
        'frames': principal_df['FRAME'].tolist(),
        'colours': principal_df['COLOUR'].tolist(),
        'legend': mapping
    }
    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(debug=True)
