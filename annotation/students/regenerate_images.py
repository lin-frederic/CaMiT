import os
import json

from test_finetuned import TestFinetuneDataset
from dataset_cropvlm import resize_transform
import numpy as np
import cv2
from tqdm import tqdm

with open("outputs/class_to_idx.json") as f:
    class_to_idx = json.load(f)
dataset = TestFinetuneDataset("outputs/test_models", class_to_idx, transform=resize_transform)

with open("outputs/annotation_scores.json") as f:
    annotation_scores = json.load(f)

test_annotations_with_box_and_score = {}

os.makedirs("outputs/regenerated_images", exist_ok=True)
for i in tqdm(range(len(dataset))):
    image, box,  score, crop_path = dataset.get_crop_with_box_and_score(i, annotation_scores)
    class_name = crop_path.split("/")[-3]
    year = crop_path.split("/")[-2]
    image_name = crop_path.split("/")[-1]
    regenerated_path = f"outputs/regenerated_images/{class_name}/{year}/{image_name}"
    test_annotations_with_box_and_score[regenerated_path] = {"box": box, "score": score}
    image = np.array(image)
    os.makedirs(f"outputs/regenerated_images/{class_name}/{year}", exist_ok=True)
    cv2.imwrite(regenerated_path, image)


with open("outputs/test_annotations_with_box_and_score.json", "w") as f:
    json.dump(test_annotations_with_box_and_score, f)



