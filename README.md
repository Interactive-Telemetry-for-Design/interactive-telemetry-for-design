![python-version](https://img.shields.io/badge/python-v3.12.8-blue)
![license](https://img.shields.io/badge/license-GPLv3-blue)
[![download](https://img.shields.io/badge/download-.zip-brightgreen)](https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design/archive/refs/heads/main.zip)

# Interactive Telemetry for Design

TODO: (description)
This project aims to enable designers to do data-driven design using active learning.
Telemetry that is collected from the usage of a designer’s prototype can be utilised to
detect anomalies. This data can give insights to designers to improve the prototype. We
will collect our own dataset for testing purposes of a bicycle using a GoPro, which has a
video stream and IMU data (accelerometer and gyroscope). The video stream is only used
to label the behaviour to train a model on the IMU data. After the model is trained, the
model can be used for anomaly detection on only new IMU data, without a video stream.
This approach not only optimises data utilisation for anomaly detection but also aligns
with ethical principles by minimizing privacy risks and environmental impact through
efficient data processing.

## Getting Started
### Dependencies
- Python 3.12.8
- CUDA 
TODO:
Clone the repository using `git clone https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design.git`
Dit doet thom

## Usage
TODO:
