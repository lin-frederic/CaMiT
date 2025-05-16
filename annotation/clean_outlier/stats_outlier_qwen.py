import os
import json
from tqdm import tqdm

splits = os.listdir("clean_outlier/outlier_qwen")
grouped_outliers = {}

for split in tqdm(splits):
    with open(f"clean_outlier/outlier_qwen/{split}") as f:
        outlier_qwen = json.load(f)
    for box_name, outlier in outlier_qwen.items():
        # outlier : ```json\n{"true": "true"}\n```
        outlier = outlier.replace("```json\n", "").replace("\n```", "").strip()
        outlier_json = json.loads(outlier)
        if box_name not in grouped_outliers:
            grouped_outliers[box_name] = outlier_json
        else:
            print(f"Duplicate box: {box_name}")
        

interiors = []
zoomed_ins = []
for box_name, box_outlier in grouped_outliers.items():
    if box_outlier["interior"]:
        interiors.append(box_name)
    if box_outlier["zoomed_in"]:
        zoomed_ins.append(box_name)

print(f"Number of interiors: {len(interiors)}")
print(f"Number of zoomed ins: {len(zoomed_ins)}")
print(f"Number of images: {len(grouped_outliers)}")
print(f"Interior ratio: {len(interiors) / len(grouped_outliers)}")
print(f"Zoomed in ratio: {len(zoomed_ins) / len(grouped_outliers)}")

with open("outputs/train_gpt_annotations_with_unknown.json") as f:
    train_annotations = json.load(f)

with open("outputs/test_gpt_annotations_with_unknown.json") as f:
    test_annotations = json.load(f)

all_annotations = {**train_annotations, **test_annotations}


import cv2
import matplotlib.pyplot as plt

plot = 1 # 0 for interiors, 1 for zoomed ins

if plot == 0:

    for box_name in interiors:
        image_id, box_id = box_name.split("_")
        box_id = int(box_id)
        image_annotation = all_annotations[image_id]
        image_path = image_annotation["image_path"].replace("/home/users/flin","/home/fredericlin")
        image = cv2.imread(image_path)
        box = image_annotation["boxes"][box_id]
        cx, cy, w, h = box["box"]
        x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.imshow(crop)
        plt.show()
    
else:
    for box_name in zoomed_ins:
        image_id, box_id = box_name.split("_")
        box_id = int(box_id)
        image_annotation = all_annotations[image_id]
        image_path = image_annotation["image_path"].replace("/home/users/flin","/home/fredericlin")
        image = cv2.imread(image_path)
        box = image_annotation["boxes"][box_id]
        cx, cy, w, h = box["box"]
        x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.imshow(crop)
        plt.show()

