![python-version](https://img.shields.io/badge/python-v3.12.8-blue)
![license](https://img.shields.io/badge/license-GPLv3-blue)
[![download](https://img.shields.io/badge/download-.zip-brightgreen)](https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design/archive/refs/heads/main.zip)

# Interactive Telemetry for Design

This project aims to enable designers to do data-driven design using active learning. 
The repository consists of two main parts, the Python code and the html website.
The model uses a 'Long, Short Term Memory' architecture to interpret long term dependencies of consumer patterns, as well as detect anomalies. 
These patterns are gathered through analysis of telemetry data; specifically, a video stream, accelerometer and gyroscope. 
This video stream is then used for active learning, as the model gets retrained based on prompting uncertain sections. 
The resulting semi-predicted timeline can then be used to give insights to designers to improve the prototype. 
The prototype is an invariant for this model: in theory any object, tool or appliance is compatible.

After the model is trained, it can be used for anomaly detection solely on the telemetry of unseen data. This approach not only optimises data utilisation for anomaly detection but also aligns with ethical principles by minimizing privacy risks and environmental impact through efficient data processing.

## Getting Started
### Dependencies
- Python 3.12.8
- See the [TensorFlow NVIDIA software requirements](https://www.tensorflow.org/install/pip#software_requirements) for GPU acceleration on Linux, optional but recommended for better performance

### Installation
Clone the repository
```
git clone https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design.git
cd interactive-telemetry-for-design
```
or download as a ZIP file.

#### Windows (CPU-based)
1. Create a virtual environment

    Using the Python launcher:
    ```
    py -3.12 -m venv .venv
    ```
    Using Python directly (ensure Python 3.12.8 is installed and available in your PATH):
    ```
    python -m venv .venv
    ```

2. Activate the environment and install the dependencies

    Using Git Bash:
    ```
    source .venv/Scripts/activate
    pip install -r requirements_win32.txt
    ```

    Using CMD:
    ```
    .venv\Scripts\activate.bat
    pip install -r requirements_win32.txt
    ```

#### Linux (GPU acceleration)

1. Create a virtual environment
    ```
    python3 -m venv .venv
    ```

2. Activate the environment and install the dependencies
    ```
    source .venv/bin/activate
    pip install -r requirements_linux.txt
    ```

### Starting the Flask server
Copy the `.env-example` file and rename the copy to `.env`. This contains the environment variable settings for the Flask server. Optionally set a secret key.

To start the Flask server, ensure your virtual environment is activated and execute
```
flask run
```
and go to [http://localhost:5000](http://localhost:5000).

## Using The Interface
In the interface, you can either upload video file in below formats, or use (pre-extracted) IMU telemetry data in a CSV file.

The model predicts labels for the whole video with associated confidence score.

Below an input threshold it becomes an anomaly.

The following formats can be uploaded for video and imu telemetry:

- GoPro (HERO 5 and later)
- Sony (a1, a7c, a7r V, a7 IV, a7s III, a9 II, a9 III, FX3, FX6, FX9, RX0 II, RX100 VII, ZV1, ZV-E10, ZV-E10 II, ZV-E1, a6700)
- Insta360 (OneR, OneRS, SMO 4k, Go, GO2, GO3, GO3S, Caddx Peanut, Ace, Ace Pro)
- DJI (Avata, Avata 2, O3 Air Unit, Action 2/4/5, Neo)
- Blackmagic RAW (*.braw)
- RED RAW (V-Raptor, KOMODO) (*.r3d)
- Freefly (Ember)
- Betaflight blackbox (*.bfl, *.bbl, *.csv)
- ArduPilot logs (*.bin, *.log)
- Gyroflow [.gcsv log](https://docs.gyroflow.xyz/app/technical-details/gcsv-format)
- iOS apps: [`Sensor Logger`](https://apps.apple.com/us/app/sensor-logger/id1531582925), [`G-Field Recorder`](https://apps.apple.com/at/app/g-field-recorder/id1154585693), [`Gyro`](https://apps.apple.com/us/app/gyro-record-device-motion-data/id1161532981), [`GyroCam`](https://apps.apple.com/us/app/gyrocam-professional-camera/id1614296781)
- Android apps: [`Sensor Logger`](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=de_AT&gl=US), [`Sensor Record`](https://play.google.com/store/apps/details?id=de.martingolpashin.sensor_record), [`OpenCamera Sensors`](https://github.com/MobileRoboticsSkoltech/OpenCamera-Sensors), [`MotionCam Pro`](https://play.google.com/store/apps/details?id=com.motioncam.pro)
- Runcam CSV (Runcam 5 Orange, iFlight GOCam GR, Runcam Thumb, Mobius Maxi 4K)
- Hawkeye Firefly X Lite CSV
- XTU (S2Pro, S3Pro)
- WitMotion (WT901SDCL binary and *.txt)
- Vuze (VuzeXR)
- KanDao (Obisidian Pro, Qoocam EGO)
- [CAMM format](https://developers.google.com/streetview/publish/camm-spec)

We use [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) for extracting the IMU telemetry from these file formats.

GT is the ground truth timeline, AI is the timeline for the predictions from the AI, Ci is the timeline for confidence score, where 0 = red, 1 = green, and AN stands for anomaly. if you click on a block in the AI timeline, you can adopt the prediction onto the ground truth timeline, but it will override any overlapping blocks annotated by the user.

### License
This project is [licensed](https://github.com/interactive-Telemetry-for-Design/interactive-telemetry-for-design/blob/main/LICENSE) under the terms of the GNU General Public License v3.0 (GPLv3).
