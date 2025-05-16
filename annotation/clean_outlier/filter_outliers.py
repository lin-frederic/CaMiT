import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def filter_class(annotations):
    new_annotations = {}
    reject_class = ["abarth","ram"]
    for image_id, annotation in tqdm(annotations.items()):
        new_boxes = []
        reject_image = False
        for box in annotation["boxes"]:
            if "gpt_class" in box:
                for reject in reject_class:
                    if reject in box["gpt_class"]:
                        reject_image = True
                        break
                if box["model_probability"] < 80 or box["car_probability"] < 80:
                    reject_image = True
                if reject_image:
                    break
                if box["gpt_class"] == "mitsubishi_lancer_evolution":
                    box["gpt_class"] = "mitsubishi_lancer"
                    
            new_boxes.append(box)
        if not reject_image:
            annotation["boxes"] = new_boxes
            new_annotations[image_id] = annotation
    return new_annotations


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to the train annotations", default="outputs/train_gpt_annotations_with_unknown.json")
    parser.add_argument("--test_annotations", type=str, help="Path to the test annotations", default="outputs/test_gpt_annotations_with_unknown.json")
    parser.add_argument("--outlier_qwen_dir", type=str, help="Path to the outlier qwen directory", default="clean_outlier/outlier_qwen")
    parser.add_argument("--clean_box", type=str, help="Path to the clean box json file", default="clean_outlier/clean_box.json")
    args = parser.parse_args()

    with open(args.clean_box) as f:
        clean_box = json.load(f)
    
    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)
    
    grouped_annotations = {**test_annotations, **train_annotations} # test then train

    outlier_qwen_files = os.listdir(args.outlier_qwen_dir)
    outlier_qwen = {}
    for split in outlier_qwen_files:
        with open(os.path.join(args.outlier_qwen_dir, split)) as f:
            outlier_qwen_split = json.load(f)
        outlier_qwen = {**outlier_qwen, **outlier_qwen_split} # merge all splits

    filtered_annotations = {}

    # first filter by clean box
    for image_id, annotation in tqdm(grouped_annotations.items()):
        drop_image = clean_box.get(annotation["image_path"], False)
        if drop_image:
            continue
        filtered_annotations[image_id] = annotation
    print(f"Original annotations: {len(grouped_annotations)}")
    print(f"Filtered annotations: {len(filtered_annotations)}")

    # then filter by outlier qwen
    for box_name, outlier in tqdm(outlier_qwen.items()):
        outlier = outlier.replace("```json\n", "").replace("\n```", "").strip()
        outlier_json = json.loads(outlier)
        if outlier_json["interior"] or outlier_json["zoomed_in"]:
            image_id, box_id = box_name.split("_")
            box_id = int(box_id)
            if image_id in filtered_annotations:
                del filtered_annotations[image_id] # remove the image if it has an outlier box


    count = 0 
    for image_id, annotation in filtered_annotations.items():
        has_known = False
        for box in annotation["boxes"]:
            if "gpt_class" in box and "unknown" not in box["gpt_class"]:
                has_known = True
                break
        count += has_known
    print(f"Total images with known classes: {count}")
    print(f"Total images: {len(grouped_annotations)}")
    exit()
    print(f"Filtered annotations after outlier qwen: {len(filtered_annotations)}")

    # filter by class
    filtered_annotations = filter_class(filtered_annotations)
    print(f"Filtered annotations after class filter: {len(filtered_annotations)}")

    # check box length distribution
    
    box_length_distribution = {}
    box_length_distribution_with_unknown = {}
    for annotation in tqdm(filtered_annotations.values()):
        box_length_with_unknown = len(annotation["boxes"])
        if box_length_with_unknown not in box_length_distribution_with_unknown:
            box_length_distribution_with_unknown[box_length_with_unknown] = 0
        box_length_distribution_with_unknown[box_length_with_unknown] += 1
        boxes = [box for box in annotation["boxes"] if "gpt_class" in box] # keep only annotated boxes
        box_length = len(boxes)
        if box_length not in box_length_distribution:
            box_length_distribution[box_length] = 0
        box_length_distribution[box_length] += 1

    box_length_distribution = {k: v for k, v in sorted(box_length_distribution.items(), key=lambda item: item[0])}
    box_length_distribution_with_unknown = {k: v for k, v in sorted(box_length_distribution_with_unknown.items(), key=lambda item: item[0])}
    print("Box length distribution with unknown:", box_length_distribution_with_unknown)
    print("Box length distribution without unknown:", box_length_distribution)

    # check class distribution
    class_distribution = {}

    for annotation in tqdm(filtered_annotations.values()):
        time = annotation["time"]
        for box in annotation["boxes"]:
            if "gpt_class" in box:
                class_name = box["gpt_class"]
                if class_name not in class_distribution:
                    class_distribution[class_name] = {}
                if time not in class_distribution[class_name]:
                    class_distribution[class_name][time] = 0
                class_distribution[class_name][time] += 1

    class_distribution = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: sum(item[1].values()), reverse=True)}
    total_pairs = 0
    valid_pairs = 0
    valid_class = 0
    valid_instances = 0
    total_instances = 0
    for class_name, time_distribution in class_distribution.items():    
        time_distribution = {k: v for k, v in sorted(time_distribution.items(), key=lambda item: item[0])}
        time_valid_pairs = []
        for time, count in time_distribution.items():
            if count >= 50:
                time_valid_pairs.append(count)
            total_pairs += 1
            total_instances += count

        if len(time_valid_pairs) >= 5:
            valid_class += 1
            valid_pairs += len(time_valid_pairs)
            valid_instances += sum(time_valid_pairs)
        else:
            print(f"Class {class_name} has less than 5 valid time")

    print(f"Valid pairs: {valid_pairs}")
    print(f"Total pairs: {total_pairs}")
    print(f"Valid ratio: {valid_pairs / total_pairs}")
    print(f"Valid class: {valid_class}")
    print(f"Total class: {len(class_distribution)}")
    print(f"Valid instances: {valid_instances}")
    print(f"Total instances: {total_instances}")
    print(f"Valid instance ratio: {valid_instances / total_instances}")

    # split final annotations into train and test based on previous split
    new_train_annotations = {}
    new_test_annotations = {}

    for image_id, annotation in filtered_annotations.items():
        if image_id in train_annotations:
            new_train_annotations[image_id] = annotation
        else:
            new_test_annotations[image_id] = annotation

    print(f"New train annotations: {len(new_train_annotations)}")
    print(f"New test annotations: {len(new_test_annotations)}")

    with open("outputs/train_gpt_annotations_no_outlier.json", "w") as f:
        json.dump(new_train_annotations, f)
    with open("outputs/test_gpt_annotations_no_outlier.json", "w") as f:
        json.dump(new_test_annotations, f)