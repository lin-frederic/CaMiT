import os
import json
import shutil
from tqdm import tqdm

with open("selected_crops.json") as f:
    crops_info = json.load(f)


annotators = ["adrian", "flin", "hammar"]
selected_images = {} # each annotation is {year: {class_name: [image_paths]}}
# group all into {class_name: [image_paths]}, don't care about year and annotator
for annotator in annotators:
    with open(f"{annotator}/selected_{annotator}.json") as f:
        annotator_selected_images = json.load(f)
    for year, class_images in annotator_selected_images.items():
        for class_name, image_paths in class_images.items():
            if class_name not in selected_images:
                selected_images[class_name] = {}
            for image_path in image_paths:
                selected_images[class_name][image_path] = crops_info[class_name][year][image_path]

new_selected_images = {}
error_rate = 0
total = 0
for class_name, image_paths in selected_images.items():
    new_selected_images[class_name] = image_paths.keys()
    image_paths = new_selected_images[class_name]
    error_rate += len(image_paths)
    total += 150
    print(f"{class_name}: {len(image_paths)}/150 ({len(image_paths)/150*100:.2f}%)")

print(f"Average error rate: {error_rate}/{total} ({error_rate/total*100:.2f}%)")

import cv2
import matplotlib.pyplot as plt
os.makedirs("error_analysis", exist_ok=True)
for class_name, image_data in selected_images.items():
    os.makedirs(f"error_analysis/{class_name}", exist_ok=True)
    for image_path, crop_info in image_data.items():
        box = crop_info["box"]
        image = cv2.imread(image_path)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"error_analysis/{class_name}/{os.path.basename(image_path)}", image)