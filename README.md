![python-version](https://img.shields.io/badge/python-v3.12.8-blue)
![license](https://img.shields.io/badge/license-GPLv3-blue)
[![download](https://img.shields.io/badge/download-.zip-brightgreen)](https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design/archive/refs/heads/main.zip)

# Interactive Telemetry for Design

This project aims to enable designers to do data-driven design using active learning. 
The repository consists of two main parts, the python code and the html website.
The model uses a 'Long, Short Term Memory' architecture to interpret long term dependencies of consumer patterns, as well as detect anomalies. 
These patterns are gathered through analysis of telemetry data; specifically, a video stream, accelerometer and gyroscope. 
This video stream is then used for active learning, as the model gets retrained based on prompting uncertain sections. 
The resulting semi-predicted timeline can then be used to give insights to designers to improve the prototype. 
The prototype is an invariant for this model: in theory any object, tool or appliance is compatible.

After the model is trained, it can be used for anomaly detection solely on the telemetry of unseen data. This approach not only optimises data utilisation for anomaly detection but also aligns with ethical principles by minimizing privacy risks and environmental impact through efficient data processing.

## Getting Started
### Dependencies
- Python 3.12.8
- CUDA 
TODO:
Clone the repository using `git clone https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design.git`
Dit doet Thom
verder nog requirements.txt erbij zetten??

## Using the interface
The first step to using our model is cloning the repository to your own device.

### Uploading types of videos
Our model is compatible with the following formats:

GoPro (HERO 5 and later)

 Sony (a1, a7c, a7r V, a7 IV, a7s III, a9 II, a9 III, FX3, FX6, FX9, RX0 II, RX100 VII, ZV1, ZV-E10, ZV-E10 II, ZV-E1, a6700)

 Insta360 (OneR, OneRS, SMO 4k, Go, GO2, GO3, GO3S, Caddx Peanut, Ace, Ace Pro)

 DJI (Avata, Avata 2, O3 Air Unit, Action 2/4/5, Neo)
 Blackmagic RAW (*.braw)

 RED RAW (V-Raptor, KOMODO) (*.r3d)

 Freefly (Ember)

 Betaflight blackbox (*.bfl, *.bbl, *.csv)

 ArduPilot logs (*.bin, *.log)

 Gyroflow .gcsv log

 iOS apps: Sensor Logger, G-Field Recorder, Gyro, GyroCam

 Android apps: Sensor Logger, Sensor Record, OpenCamera
 Sensors, MotionCam Pro

 Runcam CSV (Runcam 5 Orange, iFlight GOCam GR, Runcam Thumb, Mobius Maxi 4K)

 Hawkeye Firefly X Lite CSV

 XTU (S2Pro, S3Pro)

 WitMotion (WT901SDCL binary and *.txt)

 Vuze (VuzeXR)

 KanDao (Obisidian Pro, Qoocam EGO)

 CAMM format

In the src folder, the functions we built are separated by purpose. 
For instance, preprocessing (parsing the telemetry data into a Pandas dataframe), sequencing, anomaly detection and building the model.

The html webpage is quite self explanatory. You can either upload video file in above formats, or pre-extracted telemetry data in a csv file.

The model predicts labels for the whole video with associated confidence score.

Below an input threshold it becomes an anomaly.
