import os
import json
import argparse
import re
from unidecode import unidecode
from tqdm import tqdm

def normalize_model_name(model_name):
    model_name = model_name.lower().strip()
    model_name = unidecode(model_name)
    model_name = model_name.replace(" ", "_")

    # remove duplicate brand names (ram_ram 2500 -> ram_2500)
    model_name = re.sub(r'\b(\w+)_\1\b', r'\1', model_name)  # "audi_audi 100" → "audi_100"

    # remove "class" suffix if it appears
    #model_name = re.sub(r'\b(\w+)_class\b', r'\1', model_name)  # "mercedes_benz e_class" → "mercedes_benz e"
    #model_name = re.sub(r'\b(\w+)_cls\b', r'\1', model_name)  # "mercedes_benz e_cls" → "mercedes_benz e"
    
    model_name = model_name.replace(" ", "_")

    # standardize hyphens
    #model_name = re.sub(r'(\w)-(\d+)', r'\1\2', model_name)  # "oldsmobile_f-85" → "oldsmobile_f85"
    model_name = model_name.replace("-", "") # remove hyphens

    parts = model_name.split("_")
    if len(parts) > 2 and parts[-1] == parts[-2]:
        parts.pop()
    model_name = "_".join(parts)

    return model_name
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Qwen.")
    parser.add_argument("--annotations", type=str, help="Path to annotations JSON file", default="outputs/deduplicated_test_annotations.json")
    parser.add_argument("--output_file", type=str, help="Path to save normalized annotations", default="outputs/normalized_test_annotations.json")

    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        test_annotations = json.load(f)

    valid_classes = set()
    valid_brands = set()
    qwen_class_counts = {}
    for image_id in tqdm(test_annotations):
        image_annotations = test_annotations[image_id]
        for box in image_annotations["boxes"]:
            if box["brand"] == "unknown" or box["underrepresented"]:
                continue
            else:
                class_name = box["brand"] + "_" + box["model"]
                class_name = normalize_model_name(class_name)
                valid_classes.add(class_name)
                valid_brands.add(box["brand"])
    
    # reset underrepresented fields based on normalized class names
    changed = 0
    new_test_annotations = {}
    for image_id in test_annotations:
        image_annotations = test_annotations[image_id]  
        new_boxes = []
        for box in image_annotations["boxes"]:
            if box["brand"] == "unknown":
                new_boxes.append(box)
                continue
            else:
                class_name = box["brand"] + "_" + box["model"]
                class_name = normalize_model_name(class_name)
                if class_name in valid_classes:
                    new_boxes.append(box)
                    if box["underrepresented"]:
                        changed += 1
                        box["underrepresented"] = False
                else:
                    new_boxes.append(box)
                    box["underrepresented"] = True

        new_image_annotations = image_annotations.copy()
        new_image_annotations["boxes"] = new_boxes
        new_test_annotations[image_id] = new_image_annotations
    
    count=0
    for image_id in new_test_annotations:
        image_annotations = new_test_annotations[image_id]
        for box in image_annotations["boxes"]:
            if box["brand"]!="unknown":
                count+=1
    print(f"Changed underrepresented field for {changed} boxes.")
    print(len(test_annotations), len(new_test_annotations)) # should be the same
    with open(args.output_file, "w") as f:
        json.dump(new_test_annotations, f, indent=2)