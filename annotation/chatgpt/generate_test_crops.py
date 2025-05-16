import os
import json

import cv2
import numpy as np
from tqdm import tqdm
import argparse

def get_image(image_path, box,resize=512):
    image = cv2.imread(image_path)
    # crop image
    x1, y1, x2, y2 = box
    image = image[y1:y2, x1:x2]
    # if longest side is greater than 512, resize to 512 keeping aspect ratio
    if max(image.shape) > resize:
        scale = resize / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test crops")
    parser.add_argument("--annotations", type=str, help="Path to annotations JSON file", default="outputs/deduplicated_test_annotations.json")
    parser.add_argument("--test_crops_dir", type=str, help="Path to directory to save test crops", default="test_crops")
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        test_annotations = json.load(f)

    os.makedirs(args.test_crops_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    test_crops_count = 0

    class_mapping = {}
    for image_name in tqdm(test_annotations):
        image_annotations = test_annotations[image_name]
        image_path = image_annotations["image_path"]
        # replace home path
        image_path = image_path.replace("/home/users/flin", "/home/fredericlin")
        image = cv2.imread(image_path)
        for i, box in enumerate(image_annotations["boxes"]):
            if box["brand"] == "unknown":
                continue
            class_name = box["brand"]+"_"+box["model"]
            cx, cy, w, h = box["box"]
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            crop = get_image(image_path, (x1, y1, x2, y2))
            crop_path = os.path.join(args.test_crops_dir, f"{image_name}_{i}.jpg")
            class_mapping[crop_path] = class_name
            cv2.imwrite(crop_path, crop)

            test_crops_count += 1

    with open(os.path.join(args.test_crops_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f,
                    indent=4, sort_keys=True)
    
    print(f"Generated {test_crops_count} test crops")

    

