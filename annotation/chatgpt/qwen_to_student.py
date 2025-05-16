import os
import json
import argparse
from unidecode import unidecode
from tqdm import tqdm
def get_valid_classes(test_dir):
    classes = os.listdir(test_dir)
    classes = [c for c in classes if c != "annotations.json"]

    brand_models = {}
    for class_name in classes:
        class_parts = class_name.split("_")
        if class_parts[0] == "alfa":
            brand = "alfa_romeo"
            model = "_".join(class_parts[2:])
        elif class_parts[0] == "aston":
            brand = "aston_martin"
            model = "_".join(class_parts[2:])
        elif class_parts[0] == "land":
            brand = "land_rover"
            model = "_".join(class_parts[2:])
        else:
            brand = class_parts[0]
            model = "_".join(class_parts[1:])

        brand = unidecode(brand).lower()
        model = unidecode(model).lower()
        if brand not in brand_models:
            brand_models[brand] = set()
        brand_models[brand].add(model)
    return brand_models


def rectify_annotations(annotations, brand_models, class_mapping):
    new_annotations = {}
    for image_id in tqdm(annotations):
        image_annotations = annotations[image_id].copy()
        new_boxes = []
        for box in image_annotations["boxes"]:
            if box["brand"] == "unknown":
                new_boxes.append(box)
                continue
            else:
                brand = box["brand"].replace("-", "").replace(" ", "_")
                model = box["model"].replace("-", "")
                model = unidecode(model).lower()

                if model == "hurracan": # fix typo
                    model = "huracan"
                    class_name = "lamborghini_huracan"
                elif box["underrepresented"]:
                    assert brand in brand_models, f"Brand {brand} not found in brand_models"
                    for valid_model in brand_models[brand]:
                        if valid_model in model:
                            class_name = f"{brand}_{valid_model}" 
                            break
                    else:
                        class_name = f"{brand}_unknown"
                else:
                    class_name = f"{brand}_{model}"
                class_name = class_name.replace(" ", "_").replace("-", "").lower()
                if class_name in class_mapping:
                    box["class"] = class_name
                new_boxes.append(box)
        image_annotations["boxes"] = new_boxes
        new_annotations[image_id] = image_annotations
    return new_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Qwen scores.")
    parser.add_argument("--test_dir", type=str, help="Path to directory containing test images", default="outputs/test_gpt")
    parser.add_argument("--test_annotations", type=str, help="Path to test annotations JSON file", default="outputs/normalized_test_annotations.json")
    parser.add_argument("--train_annotations", type=str, help="Path to train annotations JSON file", default="outputs/normalized_train_annotations.json")
    parser.add_argument("--qwen_test_annotations", type=str, help="Path to test annotations JSON file with Qwen scores", default="outputs/qwen_test_annotations.json")
    parser.add_argument("--qwen_train_annotations", type=str, help="Path to train annotations JSON file with Qwen scores", default="outputs/qwen_train_annotations.json")
    parser.add_argument("--class_mapping", type=str, help="Path to class mapping JSON file", default="outputs/gpt_class_mapping.json")
    args = parser.parse_args()

    with open(args.class_mapping, "r") as f:
        class_mapping = json.load(f)

    test_brand_models = get_valid_classes(args.test_dir)


    with open(args.test_annotations, "r") as f:
        test_annotations = json.load(f)

    
    with open(args.train_annotations, "r") as f:
        train_annotations = json.load(f)

    test_annotations = rectify_annotations(test_annotations, test_brand_models, class_mapping)
    train_annotations = rectify_annotations(train_annotations, test_brand_models, class_mapping)

    with open(args.qwen_test_annotations, "w") as f:
        json.dump(test_annotations, f)

    with open(args.qwen_train_annotations, "w") as f:
        json.dump(train_annotations, f)

