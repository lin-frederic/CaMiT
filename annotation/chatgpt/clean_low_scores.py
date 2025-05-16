import os
import json
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Clean samples with low scores.")
    parser.add_argument("--parsed_results", type=str, help="Path to parsed results JSON file", default="chatgpt/parsed_results.json")
    parser.add_argument("--test_dir", type=str, help="Path to directory containing test images", default="outputs/test_gpt")

    args = parser.parse_args()

    with open(args.parsed_results, "r") as f:
        parsed_results = json.load(f)

    classes = os.listdir(args.test_dir) # include all classes dir + annotations.json
    
    with open(os.path.join(args.test_dir, "annotations.json"), "r") as f:
        annotations = json.load(f)
    
    classes = [c for c in classes if c != "annotations.json"]
    total = 0
    for class_name in classes:
        for time in os.listdir(os.path.join(args.test_dir,class_name)):
            for crop in os.listdir(os.path.join(args.test_dir,class_name,time)):
                total += 1
    with tqdm(total=total) as pbar:
        for class_name in classes:
            for time in os.listdir(os.path.join(args.test_dir,class_name)):
                for crop in os.listdir(os.path.join(args.test_dir,class_name,time)):
                    crop_path = os.path.join(args.test_dir,class_name,time,crop)
                    crop_name = os.path.basename(crop_path)
                    map_path = f"test_crops/{crop_name}"
                    crop_results = parsed_results[map_path]["gpt-4o"]
                    model_probability = crop_results["model_probability"]
                    car_probability = crop_results["car_probability"]
                    if (car_probability < 80) or (model_probability < 80):
                        annotations.pop(crop_path)
                        os.remove(crop_path)

                    pbar.update(1)
    
    with open(os.path.join(args.test_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f)
    