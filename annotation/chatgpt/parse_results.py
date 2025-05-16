import os
import json
import argparse
from tqdm import tqdm
from unidecode import unidecode

import re

# model mapping for gpt-4o
model_mapping = {
    "ford_fusion": "ford_mondeo",
    "acura_tsx": "acura_tlx", # acura_tlx is the successor of acura_tsx so they are similar
    "ford_f150": "ford_fseries",
    "mazda6": "mazda_mazda6",
    #aston_mastin_rapide

}

def normalize_model_name(model_name):
    model_name = model_name.lower().strip()
    model_name = unidecode(model_name)
    model_name = model_name.replace(" ", "_")

    # remove duplicate brand names (ram_ram 2500 -> ram_2500)
    model_name = re.sub(r'\b(\w+)_\1\b', r'\1', model_name)  # "audi_audi 100" → "audi_100"
    
    model_name = model_name.replace(" ", "_")

    """# standardize hyphens
    model_name = re.sub(r'(\w)-(\d+)', r'\1\2', model_name)  # "oldsmobile_f-85" → "oldsmobile_f85"""
    model_name = model_name.replace("-", "") # remove hyphens
    

    parts = model_name.split("_")
    if len(parts) > 2 and parts[-1] == parts[-2]:
        parts.pop()
    model_name = "_".join(parts)

    if model_name in model_mapping:
        model_name = model_mapping[model_name]

    return model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse batch results.")
    parser.add_argument("--grouped_results_file", type=str, help="Path to grouped results file", default="chatgpt/grouped_results.jsonl")
    parser.add_argument("--image_dir", type=str, help="Path to directory containing images", default="test_crops")
    parser.add_argument("--output_file", type=str, help="Path to output file", default="chatgpt/parsed_results.json")
    args = parser.parse_args()

    with open(f"{args.image_dir}/class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    
    with open(args.grouped_results_file, "r") as f:
        results = [json.loads(line) for line in f]
    parsed_results = {}

    errors = 0
    for result in tqdm(results,desc="Parsing results"):
        custom_id = result["custom_id"]
        response = result["response"]["body"]["choices"][0]["message"]["content"]
        try:
            response = json.loads(response)
            if "model" not in response:
                print(response)
                errors += 1
                continue
            response["model"] = response["model"].strip().lower()
            # remove accent
            response["model"] = unidecode(response["model"])
            response["model"] = response["model"].replace(" ", "_")
            parsed_results[custom_id] = response
        except:
            print(response)
            errors += 1
            parsed_results[custom_id] = {"model": "unknown",
                                         "model_probability": 0,
                                         "car_probability": 0}

    
    print(f"Total results: {len(results)}")
    print(f"Errors: {errors}")

    results = {}

    agreements = 0
    for custom_id in tqdm(parsed_results,desc="Creating final results"):
        image_path = f"{args.image_dir}/{custom_id}.jpg"
        class_name = class_mapping[image_path]
        class_name = normalize_model_name(class_name)
        results[image_path] = {"gpt-4o": parsed_results[custom_id],
                              "qwen": class_name}
        results[image_path]["gpt-4o"]["model"] = normalize_model_name(results[image_path]["gpt-4o"]["model"])
        if parsed_results[custom_id]["model"] in class_name or class_name in parsed_results[custom_id]["model"]:
            agreements += 1
    
    print(f"Rate of agreement: {agreements/len(parsed_results)}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    