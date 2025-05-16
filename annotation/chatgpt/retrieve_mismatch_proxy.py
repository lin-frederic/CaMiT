import os
import json
from tqdm import tqdm
from unidecode import unidecode
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Retrieve mismatched annotations.")
    parser.add_argument("--test_dir", type=str, help="Path to directory containing test images", default="outputs/test_gpt")
    parser.add_argument("--gpt_annotations_with_qwen", type=str, help="Path to GPT annotations with Qwen scores JSON file", default="outputs/gpt_annotations_with_qwen.json")

    args = parser.parse_args()

    with open(args.gpt_annotations_with_qwen, "r") as f:
        gpt_annotations = json.load(f)

    # check annotations where qwen class is different from gpt class
    mismatch = {}
    correct = {}
    for crop_name in tqdm(gpt_annotations):
        crop_annotations = gpt_annotations[crop_name]
        gpt_class = gpt_annotations[crop_name]["class_name"]
        # instead of qwen class use the proxy class
        top_classes = gpt_annotations[crop_name]["qwen_score"]["top_preds"][:2]
        top_classes = [(c[0].replace("-","").replace(" ","_").lower(), c[1]) for c in top_classes]
        qwen_class = crop_annotations['qwen_score']['top_preds'][0][0]
        qwen_class = gpt_annotations[crop_name]["qwen_class"].replace("-","").replace(" ","_").lower()
        if gpt_class not in [c[0] for c in top_classes]:
            mismatch[crop_name] = crop_annotations
        elif gpt_class != qwen_class:
            mismatch[crop_name] = crop_annotations
        else:
            class_name, time, image = crop_name.split("/")[2:]
            if class_name not in correct:
                correct[class_name] = {}
            if time not in correct[class_name]:
                correct[class_name][time] = []
            correct[class_name][time].append(image)

        """if gpt_class != qwen_class or gpt_class not in [c[0] for c in top_classes]:
            print(crop_annotations)
            exit()
            mismatch[crop_name] = crop_annotations"""
    print(f"Found {len(mismatch)}/{len(gpt_annotations)} ({len(mismatch)/len(gpt_annotations)*100:.2f}%) mismatched annotations.")
    
    # count mismatch per gpt class
    gpt_class_mismatch = {}
    for crop_name, crop_annotations in mismatch.items(): 
        gpt_class = crop_annotations["class_name"]
        if gpt_class not in gpt_class_mismatch:
            gpt_class_mismatch[gpt_class] = {}
        gpt_class_mismatch[gpt_class][crop_name] = crop_annotations
    gpt_class_mismatch = dict(sorted(gpt_class_mismatch.items(), key=lambda x: len(x[1]), reverse=True))

    # print top 5 gpt classes with most mismatches
    keys = list(gpt_class_mismatch.keys())
    for i, key in enumerate(keys[:5]):
        print(f"{i+1}. {key}: {len(gpt_class_mismatch[key])}")
        class_keys = list(gpt_class_mismatch[key].keys())
        print({k:gpt_class_mismatch[key][k] for k in class_keys[:5]})
        print()

    # among mismatched annotations, check how much is unknown
    unknown_count = 0
    for class_name in gpt_class_mismatch:
        if "unknown" in class_name:
            unknown_count += len(gpt_class_mismatch[class_name])
    print(f"Unknown count: {unknown_count}/{len(mismatch)} ({unknown_count/len(mismatch)*100:.2f}%)")
    print(f"Remaining to check: {len(mismatch)-unknown_count}/{len(gpt_annotations)} ({(len(mismatch)-unknown_count)/len(gpt_annotations)*100:.2f}%)")
    print(f"Total mismatched images: {len(mismatch)}")
    print(f"Total correct images: {sum([len(correct[class_name][time]) for class_name in correct for time in correct[class_name]])}")
    # save mismatched annotations
    with open("outputs/mismatched_annotations_with_proxy.json", "w") as f:
        json.dump(mismatch, f, indent=4)
    
    with open("outputs/correct_annotations_with_proxy.json", "w") as f:
        json.dump(correct, f, indent=4)