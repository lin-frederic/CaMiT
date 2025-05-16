import os
import json
import argparse
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)


# Argument parser for command-line options
def create_parser():
    parser = argparse.ArgumentParser(description="Occlusion Annotation App")
    parser.add_argument('--bin', type=int, default=0, help="Specify which bin to load (default is bin 0)")
    parser.add_argument('--user', type=str, required=True, help="Specify the user for annotation (e.g., user1, user2)")
    parser.add_argument('--port', type=int, default=5000, help="Specify the port to run the Flask app (default is 5000)")
    return parser

# Create parser and parse arguments
parser = create_parser()
args = parser.parse_args()

# Path to the annotation JSON file
ANNOTATION_FILE = f"annotations_{args.user}.json"

# Load existing annotations if the file exists
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)
else:
    annotations = {}



@app.route('/')
def index():
    bin_index = request.args.get("bin", str(args.bin))  # Use the command-line argument or default to 'bin 0'
    bin_folder = f"occlusion_analysis/bin_{bin_index}"
    
    if not os.path.exists(bin_folder):
        return f"Bin {bin_index} not found", 404

    images = []
    for filename in os.listdir(bin_folder):
        if filename.endswith((".jpg", ".png")):  # Only include images
            score = float(filename.split("_")[-1].replace(".jpg", ""))  # Extract score from filename
            occluded = annotations.get(filename, False)  # Load existing annotation
            images.append((filename, score, occluded))

    images.sort(key=lambda x: x[1])
    return render_template("index.html", bin_index=bin_index, bin_folder=bin_folder, images=images)

@app.route('/serve_image/<bin_folder>/<filename>')
def serve_image(bin_folder, filename):
    folder_path = os.path.join("occlusion_analysis", bin_folder)  # Correct path
    return send_from_directory(folder_path, filename, as_attachment=False)

@app.route('/annotate', methods=['POST'])
def annotate():
    global annotations
    data = request.json
    image_name = data.get("image")
    
    if image_name:
        if image_name in annotations:
            del annotations[image_name]  # Remove annotation if it exists (toggle off)
            annotated = False
        else:
            annotations[image_name] = True  # Mark as occluded (toggle on)
            annotated = True
        
        # Save to JSON file
        with open(ANNOTATION_FILE, "w") as f:
            json.dump(annotations, f, indent=4)
        
        return jsonify({"message": f"{image_name} {'marked' if annotated else 'unmarked'} as occluded", "annotated": annotated})
    
    return jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    # You can also pass the argument `bin` through the command line.
    app.run(debug=True, port=args.port)
