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
Dit doet thom

verder nog requirements.txt erbij zetten??

## Using the interface
The first step to using our model is cloning the repository to your own device.


In the src folder, the functions we built are separated by purpose. 
For instance, preprocessing (parsing the telemetry data into a Pandas dataframe), sequencing, anomaly detection and building the model.

The html webpage is quite self explanatory. You can either upload the an MP4 video file, or extracted telemetry data in a csv file.
Then, you can 



