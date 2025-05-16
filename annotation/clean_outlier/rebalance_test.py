import os
import argparse
import json
from tqdm import tqdm

def get_class_time_distribution(annotation):
    class_time_distribution = {}
    for image_id, annotation in annotation.items():
        time = annotation["time"]
        boxes = annotation["boxes"]
        image_path = annotation["image_path"]
        for i, box in enumerate(boxes):
            if "gpt_class" in box:
                class_name = box["gpt_class"]
                if class_name not in class_time_distribution:
                    class_time_distribution[class_name] = {}
                if time not in class_time_distribution[class_name]:
                    class_time_distribution[class_name][time] = {}
                if image_id not in class_time_distribution[class_name][time]:
                    class_time_distribution[class_name][time][image_id] = []
                class_time_distribution[class_name][time][image_id].append(i)
    # sort by class name
    class_time_distribution = {class_name: time_distribution for class_name, time_distribution in sorted(class_time_distribution.items(), key=lambda item: item[0])}
    # for each class, sort by time
    for class_name, time_distribution in class_time_distribution.items():
        class_time_distribution[class_name] = {time: time_images for time, time_images in sorted(time_distribution.items(), key=lambda item: int(item[0]))}
    
    return class_time_distribution

def collect_reject(train_class_time_distribution,test_class_time_distribution):
    rejected_class_time_pairs = {}
    for class_name, time_distribution in test_class_time_distribution.items():
        for time, time_images in time_distribution.items():
            if class_name not in train_class_time_distribution or time not in train_class_time_distribution[class_name]:
                train_count = 0
            else:
                train_count = len(train_class_time_distribution[class_name][time])
            test_count = len(time_images)
            reject = train_count + test_count < 80
            
            if reject:
                if class_name not in rejected_class_time_pairs:
                    rejected_class_time_pairs[class_name] = set()
                rejected_class_time_pairs[class_name].add(time)

    for class_name, time_distribution in train_class_time_distribution.items():
        for time, time_images in time_distribution.items():
            if class_name not in test_class_time_distribution or time not in test_class_time_distribution[class_name]:
                test_count = 0
            else:
                test_count = len(test_class_time_distribution[class_name][time])
            train_count = len(time_images)
            reject = train_count + test_count < 80

            if reject:
                if class_name not in rejected_class_time_pairs:
                    rejected_class_time_pairs[class_name] = set()
                rejected_class_time_pairs[class_name].add(time)
    
    return rejected_class_time_pairs

def reject_images(test_annotations,train_annotations,rejected_class_time_pairs):
    new_test_annotations = {}
    new_train_annotations = {}
    for image_id, annotation in test_annotations.items():
        time = annotation["time"]
        reject = False
        for i, box in enumerate(annotation["boxes"]):
            if "gpt_class" in box:
                class_name = box["gpt_class"]
                if class_name in rejected_class_time_pairs and time in rejected_class_time_pairs[class_name]:
                    reject = True
                    break
        if not reject:
            new_test_annotations[image_id] = annotation
    
    for image_id, annotation in train_annotations.items():
        time = annotation["time"]
        reject = False
        for i, box in enumerate(annotation["boxes"]):
            if "gpt_class" in box:
                class_name = box["gpt_class"]
                if class_name in rejected_class_time_pairs and time in rejected_class_time_pairs[class_name]:
                    reject = True
                    break
        if not reject:
            new_train_annotations[image_id] = annotation
    
    return new_test_annotations,new_train_annotations

def verify_reject(test_class_time_distribution,train_class_time_distribution):
    rejected_class_time_pairs = collect_reject(train_class_time_distribution,test_class_time_distribution)
    count_rejected = sum([len(v) for v in rejected_class_time_pairs.values()])
    print(f"Rejected class-time pairs: {count_rejected}")
    return count_rejected == 0

if __name__=="__main__":
    """
    Check if there are enough instances in train to redistribute to test
    Reject all class-time pairs with less than 80 instances in train+test, giving 30 instances to test
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to the train annotations", default="outputs/train_gpt_annotations_no_outlier.json")
    parser.add_argument("--test_annotations", type=str, help="Path to the test annotations", default="outputs/test_gpt_annotations_no_outlier.json")
    
    args = parser.parse_args()

    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)
    print(f"Original train annotations: {len(train_annotations)}")

    """# only keep images with atleast one instance with model_probability > 90 # this bias the test set with only high probability instances
    new_test_annotations = {}
    for image_id, annotation in test_annotations.items():
        has_high_prob = False
        for box in annotation["boxes"]:
            if "model_probability" in box and box["model_probability"] >= 90:
                has_high_prob = True
                break
        if has_high_prob:
            new_test_annotations[image_id] = annotation
        else:
            train_annotations[image_id] = annotation # redistribute to train
    print(f"Original test annotations: {len(test_annotations)}")
    print(f"New train annotations: {len(train_annotations)}")
    print(f"Filtered test annotations: {len(new_test_annotations)}")
    
    test_annotations = new_test_annotations"""
    while True:
        test_class_time_distribution = get_class_time_distribution(test_annotations)
        train_class_time_distribution = get_class_time_distribution(train_annotations)
        rejected_class_time_pairs = collect_reject(train_class_time_distribution,test_class_time_distribution)
        new_test_annotations,new_train_annotations = reject_images(test_annotations,train_annotations,rejected_class_time_pairs)
        test_annotations = new_test_annotations
        train_annotations = new_train_annotations
        print(f"New train annotations: {len(train_annotations)}")
        print(f"New test annotations: {len(test_annotations)}")

        test_class_time_distribution = get_class_time_distribution(test_annotations)
        train_class_time_distribution = get_class_time_distribution(train_annotations)
        if verify_reject(test_class_time_distribution,train_class_time_distribution):
            break
    print("All class-time pairs have atleast 80 instances in train+test")

    # redistribute
    test_class_time_distribution = get_class_time_distribution(test_annotations)
    train_class_time_distribution = get_class_time_distribution(train_annotations)
    test_redistributed_class_time_pairs = []
    for class_name, time_distribution in test_class_time_distribution.items():
        for time, time_images in time_distribution.items():
            if len(time_images) < 30:
                test_redistributed_class_time_pairs.append((class_name,time,len(time_images)))
    # sort so that we redistribute from the smallest class-time pairs
    test_redistributed_class_time_pairs = sorted(test_redistributed_class_time_pairs,key=lambda x: x[2])
    
    for class_name, time, count in test_redistributed_class_time_pairs:
        train_images = train_class_time_distribution[class_name][time]
        train_model_probs = [train_annotations[image_id]["boxes"][box_id]["model_probability"] for image_id in train_images for box_id in train_images[image_id]]
        print(train_model_probs)
        exit()
    exit()