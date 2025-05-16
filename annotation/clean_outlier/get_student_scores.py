import os
import json
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_scores", type=str, default="outputs/gpt_student_scores_crossval")
    parser.add_argument("--qwen_scores", type=str, default="outputs/qwen_student_scores_crossval")
    parser.add_argument("--test_gpt_scores", type=str, default="outputs/gpt_student_scores.json")
    parser.add_argument("--test_qwen_scores", type=str, default="outputs/qwen_student_scores.json")

    parser.add_argument("--train_gpt_annotations", type=str, default="outputs/train_gpt_annotations.json")
    parser.add_argument("--train_qwen_annotations", type=str, default="outputs/qwen_train_annotations.json")
    parser.add_argument("--test_gpt_annotations", type=str, default="outputs/test_gpt_annotations.json")
    parser.add_argument("--test_qwen_annotations", type=str, default="outputs/qwen_test_annotations.json")


    args = parser.parse_args()

    splits = os.listdir(args.gpt_scores)
    all_annotations = {}
    grouped_gpt_scores = {}
    for split in tqdm(splits):
        split_path = os.path.join(args.gpt_scores, split)
        with open(split_path, "r") as f:
            scores = json.load(f)
        for box_path, box_data in scores.items():
            if box_path not in grouped_gpt_scores:
                grouped_gpt_scores[box_path] = box_data
            else:
                print(f"Duplicate box path: {box_path}")
    
    with open(args.test_gpt_scores, "r") as f:
        test_gpt_scores = json.load(f)
    for box_path, box_data in tqdm(test_gpt_scores.items()):
        if box_path not in grouped_gpt_scores:
            grouped_gpt_scores[box_path] = box_data
        else:
            print(f"Duplicate box path: {box_path}")
    
    print(f"Total GPT scores: {len(grouped_gpt_scores)}")

    with open(args.train_gpt_annotations, "r") as f:
        train_gpt_annotations = json.load(f)
    with open(args.test_gpt_annotations, "r") as f:
        test_gpt_annotations = json.load(f)
    
    all_gpt_annotations = {**train_gpt_annotations, **test_gpt_annotations}
    for box_path, box_data in tqdm(grouped_gpt_scores.items()):
        box_name = box_path.split("/")[-1]
        image_id, box_id = box_name.split("_")
        if "tensor" in box_id:
            box_id = box_id.replace("tensor(", "").replace(")", "")
        box_id = int(box_id.replace(".jpg", ""))
        if image_id not in all_annotations:
            image_annotations = all_gpt_annotations[image_id]
        else:
            image_annotations = all_annotations[image_id]
        box_annotations = image_annotations["boxes"][box_id]
        box_annotations["gpt_score"] = box_data
        image_annotations["boxes"][box_id] = box_annotations
        all_annotations[image_id] = image_annotations

    grouped_qwen_scores = {}
    splits = os.listdir(args.qwen_scores)
    for split in tqdm(splits):
        split_path = os.path.join(args.qwen_scores, split)
        with open(split_path, "r") as f:
            scores = json.load(f)
        for box_path, box_data in scores.items():
            if box_path not in grouped_qwen_scores:
                grouped_qwen_scores[box_path] = box_data
            else:
                print(f"Duplicate box path: {box_path}")
        
    with open(args.test_qwen_scores, "r") as f:
        test_qwen_scores = json.load(f)

    for box_path, box_data in tqdm(test_qwen_scores.items()):
        if box_path not in grouped_qwen_scores:
            grouped_qwen_scores[box_path] = box_data
        else:
            print(f"Duplicate box path: {box_path}")
    
    with open(args.train_qwen_annotations, "r") as f:
        train_qwen_annotations = json.load(f)
    with open(args.test_qwen_annotations, "r") as f:
        test_qwen_annotations = json.load(f)
    
    all_qwen_annotations = {**train_qwen_annotations, **test_qwen_annotations}

    for box_path, box_data in tqdm(grouped_qwen_scores.items()):
        box_name = box_path.split("/")[-1]
        image_id, box_id = box_name.split("_")
        if "tensor" in box_id:
            box_id = box_id.replace("tensor(", "").replace(")", "")
        box_id = int(box_id.replace(".jpg", ""))
        if image_id not in all_annotations: # not in gpt annotations
            image_annotations = all_qwen_annotations[image_id]
        else:
            image_annotations = all_annotations[image_id]
        box_annotations = image_annotations["boxes"][box_id]
        if image_id in all_annotations:
            box_annotations["qwen_class"] = all_qwen_annotations[image_id]["boxes"][box_id]["class"]
        box_annotations["qwen_score"] = box_data
        image_annotations["boxes"][box_id] = box_annotations
        all_annotations[image_id] = image_annotations
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Total boxes: {sum([len(image_data['boxes']) for image_data in all_annotations.values()])}")
    with open("outputs/student_annotations.json", "w") as f:
        json.dump(all_annotations, f)


        
        
