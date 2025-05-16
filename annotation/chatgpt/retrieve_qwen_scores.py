import os
import json
import argparse
from tqdm import tqdm
from unidecode import unidecode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Qwen scores.")
    parser.add_argument("--test_dir", type=str, help="Path to directory containing test images", default="outputs/test_gpt")
    parser.add_argument("--original_annotations", type=str, help="Path to original annotations JSON file", default="outputs/cleaned_selected_annotations.json")
    parser.add_argument("--used_annotations", type=str, help="Path to used annotations JSON file", default="outputs/normalized_test_annotations.json")
    parser.add_argument("--qwen_scores", type=str, help="Path to save Qwen scores", default="outputs/annotation_scores.json")
    parser.add_argument("--output_file", type=str, help="Path to save normalized annotations", default="outputs/gpt_annotations_with_qwen.json")

    args = parser.parse_args()

    with open(args.original_annotations, "r") as f:
        original_annotations = json.load(f)

    with open(args.used_annotations, "r") as f:
        used_annotations = json.load(f)

    with open(args.qwen_scores, "r") as f:
        qwen_scores = json.load(f)

    # simplify image path
    new_qwen_scores = {}
    for image_path in tqdm(qwen_scores): 
        image_name = os.path.basename(image_path)
        image_parts = image_name.split("_")
        if image_name.startswith("model"):
            image_id = image_parts[1]
            box_id = image_parts[2]
        else:
            image_id = image_parts[0]
            box_id = image_parts[1]
        new_qwen_scores[image_id+"_"+box_id] = qwen_scores[image_path]


    classes = os.listdir(args.test_dir) # include all classes dir + annotations.json

    with open(os.path.join(args.test_dir, "annotations.json"), "r") as f:
        gpt_annotations = json.load(f)
    
    classes = [c for c in classes if c != "annotations.json"]

    total = 0
    for class_name in classes:
        for time in os.listdir(os.path.join(args.test_dir,class_name)):
            for crop in os.listdir(os.path.join(args.test_dir,class_name,time)):
                total += 1

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
    
    # sort brand/models
    for brand in brand_models:
        brand_models[brand] = sorted(list(brand_models[brand]))
    brand_models = {k: v for k, v in sorted(brand_models.items())}
    
    new_annotations = {}

    bad_parses = []
    with tqdm(total=total) as pbar:
        for class_name in classes:
            for time in os.listdir(os.path.join(args.test_dir,class_name)):
                for crop in os.listdir(os.path.join(args.test_dir,class_name,time)):
                    crop_path = os.path.join(args.test_dir,class_name,time,crop)
                    crop_name = os.path.basename(crop_path)
                    image_id, crop_id = crop_name.split("_")
                    crop_id = int(crop_id.split(".")[0])
                    image_annotations = used_annotations[image_id]
                    image_path = image_annotations["image_path"]
                    original_image_annotations = original_annotations[image_path]
                    box = image_annotations["boxes"][crop_id]

                    # find original box
                    for original_id, original_box in enumerate(original_image_annotations["boxes"]):
                        if original_box["box"] == box["box"]:
                            break
                    else:
                        raise ValueError("Box not found")

                    original_box["brand"] = original_box["brand"].replace("-", "").replace(" ", "_")
                    original_box["model"] = original_box["model"].replace("-", "")
                    original_box["model"] = unidecode(original_box["model"]).lower()
                    if original_box["model"] == "hurracan": # fix typo
                        original_box["model"] = "huracan"
                        original_class_name = "lamborghini_huracan"
                    elif original_box["underrepresented"]:
                        assert original_box["brand"] in brand_models, f"Brand {original_box['brand']} not found"
                        for model in brand_models[original_box["brand"]]:
                            if model in original_box["model"]: # variation of existing base model (ie citroen c3 wrx vs citroen c3)
                                original_class_name = "_".join([original_box["brand"],model]) # use the existing base model
                                bad_parses.append(original_class_name)
                                break
                        else:
                            original_class_name = "_".join([original_box["brand"], "unknown"])
                    else:
                        original_class_name = "_".join([original_box["brand"], original_box["model"]])

                    original_image_name = image_id + "_" + str(original_id)
                    qwen_score = new_qwen_scores[original_image_name]

                    new_annotations[crop_path] = gpt_annotations[crop_path].copy()
                    new_annotations[crop_path]["qwen_class"] = original_class_name
                    new_annotations[crop_path]["qwen_pred"] = "_".join([original_box["brand"], original_box["model"]])
                    new_annotations[crop_path]["qwen_score"] = qwen_score

                    pbar.update(1)
    print(f"Bad parses: {len(bad_parses)}")
    with open(args.output_file, "w") as f:
        json.dump(new_annotations, f)
                
