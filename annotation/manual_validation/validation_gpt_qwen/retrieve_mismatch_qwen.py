import os
import json
import argparse
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Retrieve mismatched annotations.")
parser.add_argument("--mismatched_annotations", type=str, help="Path to mismatched annotations JSON file", default="outputs/mismatched_annotations.json")
parser.add_argument("--images_dir", type=str, help="Path to directory containing images", default="validation_gpt_qwen/images")
parser.add_argument("--images_list", type=str, help="Path to list of images", default="validation_gpt_qwen/mismatched_images.json")

args = parser.parse_args()

with open(args.mismatched_annotations, "r") as f:
    mismatched_annotations = json.load(f)

new_mismatched_annotations = {}
os.makedirs(args.images_dir, exist_ok=True)
for crop_path in mismatched_annotations:
    crop_annotations = mismatched_annotations[crop_path]
    if "ram" in crop_annotations["class_name"] or "ram" in crop_annotations["qwen_class"]:
        continue
    if "abarth" in crop_annotations["class_name"] or "abarth" in crop_annotations["qwen_class"]:
        continue
    new_mismatched_annotations[crop_path] = crop_annotations.copy()

mismatched_sets = {}
for crop_path in tqdm(new_mismatched_annotations):
    gpt_class_name = new_mismatched_annotations[crop_path]["class_name"]

    time = crop_path.split("/")[-2]
    crop_name = crop_path.split("/")[-1]

    if gpt_class_name not in mismatched_sets:
        mismatched_sets[gpt_class_name] = {}
    if time not in mismatched_sets[gpt_class_name]:
        mismatched_sets[gpt_class_name][time] = []
    output_path = os.path.join(args.images_dir, gpt_class_name, crop_name)
    mismatched_sets[gpt_class_name][time].append((output_path, new_mismatched_annotations[crop_path]))
    os.makedirs(os.path.join(args.images_dir, gpt_class_name), exist_ok=True) 
    
    shutil.copy(crop_path, output_path)

with open(args.images_list, "w") as f:
    json.dump(mismatched_sets, f)

print(f"Total mismatched images: {len(new_mismatched_annotations)}")