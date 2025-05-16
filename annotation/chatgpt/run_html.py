from flask import Flask, render_template
import os
import argparse

parser = argparse.ArgumentParser(description="Run a Flask app to display the results")
parser.add_argument("--port", type=int, default=6006, help="Port to run the Flask app on")
parser.add_argument("--images_dir", type=str, help="Directory containing images", default="../test_crops")
args = parser.parse_args()

app = Flask(__name__,static_folder=args.images_dir)

@app.route('/')
def index():
    return render_template("results.html")

if __name__ == '__main__':
    # Make sure to run the Flask app on a specified port (e.g., 5000)
    app.run(debug=True, port=args.port)
