import os
import json
import argparse
from cv2 import norm
from tqdm import tqdm
import shutil
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check mismatch between Qwen and GPT-4o.")
    parser.add_argument("--annotations", type=str, help="Path to annotations JSON file", default="outputs/deduplicated_test_annotations.json")
    parser.add_argument("--parsed_results", type=str, help="Path to parsed results JSON file", default="chatgpt/parsed_results.json")
    parser.add_argument("--mismatch_folder", type=str, help="Path to folder containing mismatched crops", default="chatgpt/mismatched_crops")
    parser.add_argument("--mismatch_results", type=str, help="Path to save mismatched results", default="chatgpt/mismatched_results.json")
    parser.add_argument("--N", type=int, help="Number of crops to display", default=0)
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        test_annotations = json.load(f)
    
    with open(args.parsed_results, "r") as f:
        parsed_results = json.load(f)

    mismatch_crops = []
    for crop_path in tqdm(parsed_results):
        crop_results = parsed_results[crop_path]
        gpt_4o = crop_results["gpt-4o"]["model"]
        qwen = crop_results["qwen"]
        # instead of equality, check if they are substrings of each other
        if gpt_4o in qwen or qwen in gpt_4o:
            continue
        mismatch_crops.append(crop_path)
    underrepresented_mismatch = []
    for crop_path in tqdm(mismatch_crops):
        crop_name = os.path.basename(crop_path).split(".")[0]
        image_id, crop_id = crop_name.split("_")
        image_annotations = test_annotations[image_id]
        crop_annotation = image_annotations["boxes"][int(crop_id)]
        if crop_annotation["underrepresented"]:
            underrepresented_mismatch.append(crop_path)
    
    print(f"Total mismatch: {len(mismatch_crops)}")
    print(f"Ratio of agreement: {1-len(mismatch_crops)/len(parsed_results)}")
    print(f"Underrepresented mismatch: {len(underrepresented_mismatch)}")
    print(f"Ratio: {len(underrepresented_mismatch)/len(mismatch_crops)}")

    # check class distribution for mismatched crops
    qwen_class_counts = {}
    for crop_path in parsed_results: # get real distribution for qwen
        qwen_class = parsed_results[crop_path]["qwen"]
        if qwen_class not in qwen_class_counts:
            qwen_class_counts[qwen_class] = 0
        qwen_class_counts[qwen_class] += 1


    gpt_4o_class_counts = {}

    for crop_path in mismatch_crops: # get mismatched distribution for gpt-4o
        gpt_4o_class = parsed_results[crop_path]["gpt-4o"]["model"]

        if gpt_4o_class not in gpt_4o_class_counts:
            gpt_4o_class_counts[gpt_4o_class] = 0
        gpt_4o_class_counts[gpt_4o_class] += 1

    qwen_class_counts = {k: v for k, v in sorted(qwen_class_counts.items(), key=lambda item: item[1], reverse=True)}

    gpt_4o_class_counts = {k: v for k, v in sorted(gpt_4o_class_counts.items(), key=lambda item: item[1], reverse=True)}
    
    not_in_qwen = []

    for gpt4o_class in gpt_4o_class_counts:
        for qwen_class in qwen_class_counts:
            if gpt4o_class in qwen_class or qwen_class in gpt4o_class:
                break
        else:
            not_in_qwen.append(gpt4o_class)
    
    not_in_qwen_counts = {k: gpt_4o_class_counts[k] for k in not_in_qwen}
    not_in_qwen_counts = {k: v for k, v in sorted(not_in_qwen_counts.items(), key=lambda item: item[1], reverse=True)}
    
  
    print(sum(not_in_qwen_counts.values()))
    print(f"Ratio of classes not in Qwen: {sum(not_in_qwen_counts.values())/len(parsed_results)}")
    print(not_in_qwen_counts)


    



    if args.N>0:
        os.makedirs(args.mismatch_folder, exist_ok=True)
        # Randomly select N crops to display
        mismatch_crops = random.sample(mismatch_crops, args.N)
        mismatch_results = {}
        for crop_path in mismatch_crops:
            crop_name = os.path.basename(crop_path)
            new_crop_path = os.path.join(args.mismatch_folder, crop_name)
            shutil.copy(crop_path, new_crop_path)
            mismatch_results[new_crop_path] = parsed_results[crop_path]
            if crop_path in underrepresented_mismatch:
                mismatch_results[new_crop_path]["qwen"] += " (unknown)"

        with open(args.mismatch_results, "w") as f:
            json.dump(mismatch_results, f, indent=4)

        

