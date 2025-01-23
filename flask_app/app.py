from flask import Flask, render_template

app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Data upload page route
@app.route('/data_upload')
def data_upload():
    return render_template('data_upload.html')

# Testing page route
@app.route('/testing')
def testing():
    return render_template('testing.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)