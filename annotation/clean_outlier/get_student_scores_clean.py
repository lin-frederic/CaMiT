import os
import json
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, default="outputs/train_gpt_annotations_no_outlier.json")
    parser.add_argument("--test_annotations", type=str, default="outputs/test_gpt_annotations_no_outlier.json")
    parser.add_argument("--student_annotations", type=str, default="outputs/student_annotations.json")

    args = parser.parse_args()

    with open(args.train_annotations, "r") as f:
        train_annotations = json.load(f)
    with open(args.test_annotations, "r") as f:
        test_annotations = json.load(f)
    
    all_annotations = {**train_annotations, **test_annotations}

    with open(args.student_annotations, "r") as f:
        student_annotations = json.load(f)
    
    for image_id, annotation in tqdm(all_annotations.items()):
        if image_id not in student_annotations: # missing student annotations, should be unknown classes
            has_unknown = False
            for box in annotation["boxes"]:
                if "gpt_class" in box and "unknown" in box["gpt_class"]:
                    has_unknown = True
                    break
            if not has_unknown: # if no exit, then there are only unknown classes
                print("No unknown class")
                print(annotation)
                exit()
        else:
            student_annotation = student_annotations[image_id]
            new_boxes = []
            for box1 in annotation["boxes"]:
                for box2 in student_annotation["boxes"]:
                    if box1["box"] == box2["box"]:
                        new_boxes.append(box2)
                        break
            annotation["boxes"] = new_boxes
    
    with open("outputs/student_annotations_clean.json", "w") as f:
        json.dump(all_annotations, f, indent=4)
    
