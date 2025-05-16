import os
import json
import argparse
import shutil
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Retrieve mismatched annotations.")
parser.add_argument("--manual_annotations", type=str, help="Path to manual annotations JSON file", default="outputs/manual_annotations.json")
parser.add_argument("--original_images_dir", type=str, help="Path to directory containing original images", default="validation_test_set/images")
parser.add_argument("--images_dir", type=str, help="Path to directory containing images", default="validation_gpt/static/mismatched_images")
parser.add_argument("--images_list", type=str, help="Path to list of images", default="validation_gpt/mismatched_images.json")

args = parser.parse_args()

with open(args.manual_annotations, "r") as f:
    manual_annotations = json.load(f)

new_manual_annotations = {}

for image_path in manual_annotations:
    image_annotations = manual_annotations[image_path]
    if "ram" in image_annotations["class_name"] or "ram" in image_annotations["qwen_class"]:
        continue
    if "abarth" in image_annotations["class_name"] or "abarth" in image_annotations["qwen_class"]:
        continue
    new_manual_annotations[image_path] = image_annotations
print(f"Total images: {len(manual_annotations)}")
print(f"Total images without ram or abarth: {len(new_manual_annotations)}")
manual_annotations = new_manual_annotations
mismatched_images = []
for image_path in manual_annotations:
    image_annotations = manual_annotations[image_path]
    if image_annotations["class_name"] != image_annotations["qwen_class"]:
        mismatched_images.append(image_path)
print(f"Total mismatched images: {len(mismatched_images)}")

os.makedirs(args.images_dir, exist_ok=True)

mismatched_sets = {}
for image_path in tqdm(mismatched_images):
    gpt_class_name = manual_annotations[image_path]["class_name"]
    qwen_class_name = manual_annotations[image_path]["qwen_class"]
    gpt_pred = manual_annotations[image_path]["pred"]
    qwen_pred = manual_annotations[image_path]["qwen_pred"]
    time = manual_annotations[image_path]["time"]
    # retrive box and scores for image
    image_name = image_path.split("/")[-1]
    output_path = os.path.join(args.images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist")
        continue
    gpt_msg = f"{gpt_class_name} \n({gpt_pred})"
    qwen_msg = f"{qwen_class_name} \n({qwen_pred})"
    mismatched_sets[output_path] = {"gpt_class": gpt_msg,"qwen_class": qwen_msg, "time": time, "box": manual_annotations[image_path]["box"]}
    shutil.copy(image_path, output_path)
    #output_path = os.path.join(args.images_dir, gpt_class_name, image_name)

with open(args.images_list, "w") as f:
    json.dump(mismatched_sets, f)

print(f"Total mismatched images: {len(mismatched_images)}")