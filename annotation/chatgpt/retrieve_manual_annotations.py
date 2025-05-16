import os
import json
import argparse
from tqdm import tqdm
from unidecode import unidecode

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Retrieve manual annotations.")
    parser.add_argument("--test_dir", type=str, help="Path to directory containing test images", default="outputs/test_gpt")
    parser.add_argument("--gpt_annotations_with_qwen", type=str, help="Path to GPT annotations with Qwen scores JSON file", default="outputs/gpt_annotations_with_qwen.json") 
    parser.add_argument("--original_annotations", type=str, help="Path to original annotations JSON file", default="outputs/cleaned_selected_annotations.json")
    parser.add_argument("--used_annotations", type=str, help="Path to used annotations JSON file", default="outputs/normalized_test_annotations.json")
    parser.add_argument("--manual_annotations_dir", type=str, help="Path to directory containing manual annotations", default="validation_test_set")
    # users = ["adrian","flin","hammar"] , type list
    parser.add_argument("--annotators", type=str, help="Annotators to retrieve manual annotations from", default="adrian,flin,hammar")
    parser.add_argument("--output_file", type=str, help="Path to save manual annotations", default="outputs/manual_annotations.json")

    args = parser.parse_args()

    with open(args.original_annotations, "r") as f:
        original_annotations = json.load(f)

    with open(args.used_annotations, "r") as f:
        used_annotations = json.load(f)

    """with open(os.path.join(args.test_dir, "annotations.json"), "r") as f:
        gpt_annotations = json.load(f)"""
    with open(args.gpt_annotations_with_qwen, "r") as f:
        gpt_annotations = json.load(f)
    
    # in original_annotations, replace image path with image id
    original_annotations = {os.path.basename(image_path).split(".")[0]: original_annotations[image_path] for image_path in original_annotations}
    new_gpt_annotations = {}
    for box_path in gpt_annotations:
        box_parts = box_path.split("/")
        class_name = box_parts[-3]
        time = box_parts[-2]
        crop_name = box_parts[-1]
        new_gpt_annotations[crop_name] = {"time": time,
                                          "box": gpt_annotations[box_path]["box"],
                                          "pred": gpt_annotations[box_path]["pred"],
                                          "class_name": gpt_annotations[box_path]["class_name"],
                                          "qwen_class": gpt_annotations[box_path]["qwen_class"],
                                          "qwen_pred": gpt_annotations[box_path]["qwen_pred"],
                                          "qwen_score": gpt_annotations[box_path]["qwen_score"]}
        
    gpt_annotations = new_gpt_annotations
    annotators = args.annotators.split(",")

    total = 0
    for annotator in annotators:
        with open(os.path.join(args.manual_annotations_dir, f"{annotator}/not_selected_{annotator}.json"), "r") as f:
            annotator_annotations = json.load(f)
        for time in annotator_annotations:
            for class_name in annotator_annotations[time]:
                total += len(annotator_annotations[time][class_name])

    manual_annotations = {}
    filtered_count = 0 # count of boxes that are not in gpt_annotations
    mismatch = []
    with tqdm(total=total) as pbar:
        for annotator in annotators:
            with open(os.path.join(args.manual_annotations_dir, f"{annotator}/not_selected_{annotator}.json"), "r") as f:
                annotator_annotations = json.load(f)
            for time in annotator_annotations:
                for class_name in annotator_annotations[time]:
                    for box_path in annotator_annotations[time][class_name]:
                        box_name = os.path.basename(box_path)
                        box_parts = box_name.split("_")
                        if box_name.startswith("model"):
                            image_id = box_parts[1]
                            box_id = box_parts[2]
                        else:
                            image_id = box_parts[0]
                            box_id = box_parts[1]

                        image_annotations = original_annotations[image_id]
                        used_image_annotations = used_annotations[image_id]
                        box = image_annotations["boxes"][int(box_id)]
                        # map to used annotations
                        for used_box_id, used_box in enumerate(used_image_annotations["boxes"]):
                            if used_box["box"] == box["box"]:
                                box_id = used_box_id
                                break
                        #class_name = unidecode(class_name)
                        crop_name = f"{image_id}_{box_id}.jpg"
                        if crop_name not in gpt_annotations:
                            filtered_count += 1
                            pbar.update(1)
                            continue
                        gpt_crop = gpt_annotations[crop_name]
                        gpt_class_name = gpt_crop["class_name"]
                        class_name = gpt_crop["qwen_class"]
                        # check mismatch
                        if unidecode(class_name) not in unidecode(gpt_class_name):
                            print(f"Mismatch: {gpt_class_name} ({gpt_crop['pred']}) vs {class_name} ({gpt_crop['qwen_pred']})")
                            mismatch.append(box_path)
                        manual_annotations[box_path] = {"time": time,
                                                        "box": gpt_crop["box"],
                                                        "pred": gpt_crop["pred"],
                                                        "class_name": gpt_class_name,
                                                        "qwen_class": class_name,
                                                        "qwen_pred": gpt_crop["qwen_pred"],
                                                        "qwen_score": gpt_crop["qwen_score"]}

                        pbar.update(1)
    print(f"Total: {total}")
    print(f"Filtered: {filtered_count}")    
    print(f"Mismatch: {len(mismatch)}")
    with open(args.output_file, "w") as f:
        json.dump(manual_annotations, f)

