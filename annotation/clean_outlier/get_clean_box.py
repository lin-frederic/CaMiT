import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_box", type=str, help="Path to the clean box json file", default="clean_outlier/clean_box.json")
    parser.add_argument("--train_annotations", type=str, help="Path to the train annotations", default="outputs/train_gpt_annotations_with_unknown.json")
    parser.add_argument("--test_annotations", type=str, help="Path to the test annotations", default="outputs/test_gpt_annotations_with_unknown.json")
    args = parser.parse_args()

    with open(args.clean_box) as f:
        clean_box = json.load(f)
    
    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)

    grouped_annotations = {**test_annotations, **train_annotations} # test then train
    filtered_annotations = {}
    for image_id, annotation in tqdm(grouped_annotations.items()):
        drop_image = clean_box.get(annotation["image_path"], False)
        if drop_image:
            continue
        filtered_annotations[image_id] = annotation
    print(f"Original annotations: {len(grouped_annotations)}")
    print(f"Filtered annotations: {len(filtered_annotations)}")


    # check class distribution
    class_distribution = {}
    box_length_distribution = {}
    for annotation in tqdm(filtered_annotations.values()):
        boxes = [box for box in annotation["boxes"] if "gpt_class" in box]
        box_length = len(boxes)
        time = annotation["time"]
        if box_length not in box_length_distribution:
            box_length_distribution[box_length] = 0
        box_length_distribution[box_length] += 1
        for box in boxes:
            class_name = box["gpt_class"]
            if class_name not in class_distribution:
                class_distribution[class_name] = {}
            if time not in class_distribution[class_name]:
                class_distribution[class_name][time] = 0
            class_distribution[class_name][time] += 1
    
    class_distribution = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    new_class_distribution = {}

    for class_name, time_distribution in class_distribution.items():
        time_distribution = {k: v for k, v in sorted(time_distribution.items(), key=lambda item: item[0])}
        new_time_distribution = {}
        for time, count in time_distribution.items():
            if count >= 50:
                new_time_distribution[time] = count
        if len(new_time_distribution) < 5: # at least 5 time with 30+ count
            print(f"Skip class {class_name} with time distribution {new_time_distribution}")
            continue
        new_class_distribution[class_name] = new_time_distribution

    class_names = list(new_class_distribution.keys())
    print(sorted(class_names))