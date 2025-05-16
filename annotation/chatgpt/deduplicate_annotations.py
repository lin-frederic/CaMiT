import os
import json
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_iou(box1,box2):
    # box1 = [cx1, cy1, w1, h1]
    # box2 = [cx2, cy2, w2, h2]    

    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1_min, x1_max, y1_min, y1_max = cx1 - w1/2, cx1 + w1/2, cy1 - h1/2, cy1 + h1/2
    x2_min, x2_max, y2_min, y2_max = cx2 - w2/2, cx2 + w2/2, cy2 - h2/2, cy2 + h2/2

    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = w1 * h1
    box2Area = w2 * h2

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate annotations")
    parser.add_argument("--annotations", type=str, help="Path to annotations JSON file", default="outputs/cleaned_test_annotations.json")
    parser.add_argument("--output_annotations", type=str, help="Path to output annotations JSON file", default="outputs/deduplicated_test_annotations.json")
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        annotations = json.load(f)

    deduplicated_annotations = {}
    for image_name in tqdm(annotations):
        image_annotations = annotations[image_name]
        boxes = image_annotations["boxes"]
        is_duplicate = [False] * len(boxes)
        for i in range(len(boxes)):
            if is_duplicate[i]:
                continue
            for j in range(i+1, len(boxes)):
                if is_duplicate[j]:
                    continue
                iou = calculate_iou(boxes[i]["box"], boxes[j]["box"])
                if iou > 0.8:
                    is_duplicate[j] = True

        deduplicated_boxes = [boxes[i] for i in range(len(boxes)) if not is_duplicate[i]]
        # keep the same keys/values as the original annotations but with deduplicated boxes
        deduplicated_annotations[image_name] = {"time": image_annotations["time"],
                                                "boxes": deduplicated_boxes,
                                                "image_path": image_annotations["image_path"]}
        
    with open(args.output_annotations, "w") as f:
        json.dump(deduplicated_annotations, f, indent=4)
        