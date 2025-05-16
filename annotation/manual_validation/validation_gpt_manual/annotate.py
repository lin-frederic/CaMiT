from flask import Flask, render_template, request, jsonify
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Annotation App with User-specific files")
parser.add_argument("--port", type=int, default=5000, help="Port to run the app on")
parser.add_argument("--user", type=str, required=True, help="Annotator name")
args = parser.parse_args()


app = Flask(__name__)

MISMATCHED_JSON = "mismatched_images2.json"
ANNOTATIONS_JSON = f"annotation_results_{args.user}.json"
#ANNOTATIONS_JSON = "annotation_results.json"

# Load mismatched images
if os.path.exists(MISMATCHED_JSON):
    with open(MISMATCHED_JSON, "r") as f:
        mismatched_data = json.load(f)
else:
    mismatched_data = {}

# Ensure image paths match static folder
new_mismatched_data = {}
for image_path in mismatched_data:
    image_data = mismatched_data[image_path]
    new_image_path = image_path.replace("validation_gpt", "static")
    new_mismatched_data[new_image_path] = image_data
mismatched_data = new_mismatched_data


# Initialize annotation file with gpt_correct=True if it does not exist
if not os.path.exists(ANNOTATIONS_JSON):
    annotations = {image_path: {
        "gpt_class": data["gpt_class"],
        "qwen_class": data["qwen_class"],
        "time": data["time"],
        "gpt_correct": True  # Default to True
    } for image_path, data in mismatched_data.items()}

    with open(ANNOTATIONS_JSON, "w") as f:
        json.dump(annotations, f, indent=4)
else:
    with open(ANNOTATIONS_JSON, "r") as f:
        annotations = json.load(f)

@app.route("/")
def index():
    sorted_mismatched_data = dict(sorted(mismatched_data.items(),key=lambda item: item[1]["qwen_class"]))
    return render_template("index.html", mismatched_data=sorted_mismatched_data, annotations=annotations)

@app.route("/update_annotation", methods=["POST"])
def update_annotation():
    data = request.json
    image_path = data["image_path"]
    gpt_correct = data["gpt_correct"]

    # Update annotations
    annotations[image_path] = {
        "gpt_class": mismatched_data[image_path]["gpt_class"],
        "qwen_class": mismatched_data[image_path]["qwen_class"],
        "time": mismatched_data[image_path]["time"],
        "gpt_correct": gpt_correct
    }

    # Save updated annotations
    with open(ANNOTATIONS_JSON, "w") as f:
        json.dump(annotations, f, indent=4)

    return jsonify({"message": "Annotation updated", "success": True})

if __name__ == "__main__":
    app.run(debug=True, port=args.port)
