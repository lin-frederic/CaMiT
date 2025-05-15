import os
import json
from collections import defaultdict
import argparse

def count_images_in_folder(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def create_json_from_test_images(test_images_dir):
    without_year_dir = os.path.join(test_images_dir, "without_year")
    with_year_dir = os.path.join(test_images_dir, "with_year")

    model_counts_without_year = []
    model_counts_with_year = []

    # Handle WITHOUT year
    for model_name in sorted(os.listdir(without_year_dir)):
        model_path = os.path.join(without_year_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        count = count_images_in_folder(model_path)
        model_prompt = f"A photo of {model_name.replace('_', ' ')}"
        model_counts_without_year.append({
            "prompt": model_prompt,
            "count": count
        })


    for model_name in sorted(os.listdir(with_year_dir)):
        model_path = os.path.join(with_year_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        for year in sorted(os.listdir(model_path)):
            year_path = os.path.join(model_path, year)
            if not os.path.isdir(year_path):
                continue
            count = count_images_in_folder(year_path)
            model_prompt = f"A photo of {model_name.replace('_', ' ')} in {year}"
            model_counts_with_year.append({
                "prompt": model_prompt,
                "model": model_name.replace('_', ' '),
                "year": year,
                "count": count
            })


    with open("caption_counts_without_year.json", "w") as f:
        json.dump(model_counts_without_year, f, indent=4)

    with open("caption_counts_with_year.json", "w") as f:
        json.dump(model_counts_with_year, f, indent=4)

    print("✅ JSON files generated: caption_counts_without_year.json & caption_counts_with_year.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON caption counts for test images.")
    parser.add_argument("--base_images_dir", type=str, required=True, help="Path to the base directory containing test images.")
    args = parser.parse_args()

    create_json_from_test_images(args.base_images_dir)


