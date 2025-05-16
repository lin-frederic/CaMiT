import os
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

normalized_train_annotations = "outputs/normalized_train_annotations.json"
normalized_test_annotations = "outputs/normalized_test_annotations.json"
train_mapped_results = "train_mapped_results.json"
test_mapped_results = "chatgpt/mapped_results.json"

with open(train_mapped_results, "r") as f:
    train_mapped_results = json.load(f)

with open(test_mapped_results, "r") as f:
    test_mapped_results = json.load(f)

with open(normalized_train_annotations, "r") as f:
    normalized_train_annotations = json.load(f)

with open(normalized_test_annotations, "r") as f:
    normalized_test_annotations = json.load(f)

import random

test_class_counts = {}
# replace paths with {image_id}_{box_id}
test_mapped_results = {crop_path.split("/")[-1].split(".")[0]: test_mapped_results[crop_path] for crop_path in test_mapped_results}
test_gpt_annotations = {}
count = 0
for crop_path in tqdm(test_mapped_results):
    image_id, box_id = crop_path.split("_")
    box_id = int(box_id)
    image_annotations =  test_gpt_annotations.get(image_id, normalized_test_annotations[image_id].copy()) # in case there are multiple boxes for the same image 
    box = image_annotations["boxes"][box_id]
    gpt_class = test_mapped_results[crop_path]['class']
    car_probability = test_mapped_results[crop_path]['car_probability']
    model_probability = test_mapped_results[crop_path]['model_probability']
    if "unknown" in gpt_class: # skip unknown classes
        count += 1
        continue
    # if an image has at least one box with a gpt class, copy all boxes of the image
    box["gpt_class"] = gpt_class # modify the box
    box["car_probability"] = car_probability
    box["model_probability"] = model_probability
    image_annotations["boxes"][box_id] = box
    test_gpt_annotations[image_id] = image_annotations
    test_class_counts[gpt_class] = test_class_counts.get(gpt_class, 0) + 1
print(f"Skipped {count} unknown classes")

train_class_counts = {}
train_mapped_results = {crop_path.split("/")[-1].split(".")[0]: train_mapped_results[crop_path] for crop_path in train_mapped_results}

gpt_annotations = {}
for crop_name in tqdm(train_mapped_results):
    image_id, box_id = crop_name.split("_")
    box_id = int(box_id)
    image_annotations = gpt_annotations.get(image_id, normalized_train_annotations[image_id].copy()) # in case there are multiple boxes for the same image
    box = image_annotations["boxes"][box_id]
    gpt_class = train_mapped_results[crop_name]['class']
    car_probability = train_mapped_results[crop_name]['car_probability']
    model_probability = train_mapped_results[crop_name]['model_probability']
    if "unknown" in gpt_class:
        continue
    # if an image has at least one box with a gpt class, copy all boxes of the image
    box["gpt_class"] = gpt_class # modify the box
    box["car_probability"] = car_probability
    box["model_probability"] = model_probability
    image_annotations["boxes"][box_id] = box
    gpt_annotations[image_id] = image_annotations
    train_class_counts[gpt_class] = train_class_counts.get(gpt_class, 0) + 1

print(f"Test class counts ({len(test_class_counts)}):", {k: v for k, v in sorted(test_class_counts.items(), key=lambda item: item[1], reverse=True)})
print()
print(f"Train class counts ({len(train_class_counts)}):", {k: v for k, v in sorted(train_class_counts.items(), key=lambda item: item[1], reverse=True)})

# check that they contain the same classes
assert set(test_class_counts.keys()) == set(train_class_counts.keys())

class_mapping = {k: i for i, k in enumerate(sorted(test_class_counts.keys()))}
print("Class mapping:", class_mapping)
with open("outputs/train_gpt_annotations.json", "w") as f:
    json.dump(gpt_annotations, f)
with open("outputs/test_gpt_annotations.json", "w") as f:
    json.dump(test_gpt_annotations, f)
with open("outputs/gpt_class_mapping.json", "w") as f:
    json.dump(class_mapping, f)