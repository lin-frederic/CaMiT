import os
import json
import argparse
from tqdm import tqdm
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_annotations", type=str, help="Path to the cleaned annotations", default="validate_score/cleaned_annotations.json")
    parser.add_argument("--citroen_ds_annotations", type=str, help="Path to the citroen ds annotations", default="validate_score/citroen_ds_annotations.json")
    parser.add_argument("--test_images", type=str, help="Path to the test images", default="test_images")
    args = parser.parse_args()

    cleaned_annotations_path = args.cleaned_annotations
    test_images_path = args.test_images

    test_images = {}
    for split_file in tqdm(os.listdir(test_images_path)):
        if split_file == "test_images_citroen_ds.json":
            continue
        split_path = os.path.join(test_images_path, split_file)
        with open(split_path, "r") as f:
            split_data = json.load(f)
        
        for year in split_data:
            if year not in test_images:
                test_images[year] = []
            test_images[year].extend(split_data[year])
    

    with open(cleaned_annotations_path, "r") as f:
        cleaned_data = json.load(f)
    
    test_annotations = {}
    for year in test_images:
        for image_id in test_images[year]:
            test_annotations[image_id] = cleaned_data[image_id]
    
    with open(args.citroen_ds_annotations, "r") as f:
        citroen_ds_data = json.load(f)
    
    with open(os.path.join(test_images_path, "test_images_citroen_ds.json"), "r") as f:
        test_images_citroen_ds = json.load(f)
    
    for year in test_images_citroen_ds:
        for image_id in test_images_citroen_ds[year]:
           test_annotations[image_id] = citroen_ds_data[image_id]

    # Check class distribution

    class_distribution = {}
    for image_id, annotation in test_annotations.items():
        time = annotation["time"]
        for box in annotation["boxes"]:
            assert "class" in box, f"Missing class in annotation for image {image_id}"
            class_name = box["class"]
            if class_name == "unknown":
                continue
            if class_name not in class_distribution:
                class_distribution[class_name] = {}
            if time not in class_distribution[class_name]:
                class_distribution[class_name][time] = 0
            class_distribution[class_name][time] += 1
    
    # sort by class
    class_distribution = dict(sorted(class_distribution.items(), key=lambda item: item[0]))
    # sort by time
    for class_name in class_distribution:
        class_distribution[class_name] = dict(sorted(class_distribution[class_name].items(), key=lambda item: int(item[0])))
    print("Number of classes:", len(class_distribution))
    #print("Class distribution:")
    not_enough = {}
    for class_name, time_distribution in class_distribution.items():
        #print(f"{class_name}: {time_distribution}")
        for time, count in time_distribution.items():
            if count < 30:
                if class_name not in not_enough:
                    not_enough[class_name] = {}
                not_enough[class_name][time] = count
    if len(not_enough) == 0:
        print("All classes have enough samples")
    else:
        print("Not enough samples:")
        for class_name, time_distribution in not_enough.items():
            time_distribution = dict(sorted(time_distribution.items(), key=lambda item: int(item[0])))
            print(f"{class_name}: {time_distribution}")
        # for class-time in sampled test set, check that there are not enough in the original set
        # to do so, get original class distribution
        original_class_distribution = {}
        for image_id, annotation in cleaned_data.items():
            time = annotation["time"]
            for box in annotation["boxes"]:
                assert "class" in box, f"Missing class in annotation for image {image_id}"
                class_name = box["class"]
                if class_name == "unknown":
                    continue
                if class_name not in original_class_distribution:
                    original_class_distribution[class_name] = {}
                if time not in original_class_distribution[class_name]:
                    original_class_distribution[class_name][time] = 0
                original_class_distribution[class_name][time] += 1
    
    # split train and test
    train_annotations = {}
    for image_id, annotation in cleaned_data.items():
        if image_id not in test_annotations:
            train_annotations[image_id] = annotation
    
    print("Number of train annotations:", len(train_annotations))
    print("Number of test annotations:", len(test_annotations))
    print("Number of original annotations:", len(cleaned_data))

    # check there are no overlaps between train and test
    for image_id in train_annotations:
        if image_id in test_annotations:
            print(f"Overlap found: {image_id}")

    # check class distribution in train set
    train_class_distribution = {}
    train_box_length_distribution = {}
    train_gpt_scores = []
    train_qwen_scores = []
    for image_id, annotation in train_annotations.items():
        time = annotation["time"]
        box_count = 0
        for box in annotation["boxes"]:
            assert "class" in box, f"Missing class in annotation for image {image_id}"
            class_name = box["class"]
            if class_name == "unknown":
                continue
            if class_name not in train_class_distribution:
                train_class_distribution[class_name] = {}
            if time not in train_class_distribution[class_name]:
                train_class_distribution[class_name][time] = 0
            train_class_distribution[class_name][time] += 1
            box_count += 1
            train_gpt_scores.append(box["gpt_score"]["annotation_score"][1])
            train_qwen_scores.append(box["qwen_score"]["annotation_score"][1])
        if box_count not in train_box_length_distribution:
            train_box_length_distribution[box_count] = 0
        train_box_length_distribution[box_count] += 1
            
    # get min, max, avg count for each class-time
    min_count = float("inf")
    max_count = 0
    avg_count = 0
    total = 0
    for class_name, time_distribution in train_class_distribution.items():
        for time, count in time_distribution.items():
            if count < min_count:
                min_count = count
            if count > max_count:
                max_count = count
            total += 1
            avg_count += count
    avg_count /= total
    print("Min count:", min_count)
    print("Max count:", max_count)
    print("Avg count:", avg_count)
    print("Total count:", total)

    print("Box length distribution:")
    for box_length, count in train_box_length_distribution.items():
        print(f"{box_length}: {count}")
    
    # check class distribution in test set
    test_class_distribution = {}
    test_box_length_distribution = {}
    test_gpt_scores = []
    test_qwen_scores = []
    for image_id, annotation in test_annotations.items():
        time = annotation["time"]
        box_count = 0
        for box in annotation["boxes"]:
            assert "class" in box, f"Missing class in annotation for image {image_id}"
            class_name = box["class"]
            if class_name == "unknown":
                continue
            if class_name not in test_class_distribution:
                test_class_distribution[class_name] = {}
            if time not in test_class_distribution[class_name]:
                test_class_distribution[class_name][time] = 0
            test_class_distribution[class_name][time] += 1
            box_count += 1
            test_gpt_scores.append(box["gpt_score"]["annotation_score"][1])
            test_qwen_scores.append(box["qwen_score"]["annotation_score"][1])
        if box_count not in test_box_length_distribution:
            test_box_length_distribution[box_count] = 0
        test_box_length_distribution[box_count] += 1


    # get min, max, avg count for each class-time
    min_count = float("inf")
    max_count = 0
    avg_count = 0
    total = 0
    for class_name, time_distribution in test_class_distribution.items():
        for time, count in time_distribution.items():
            if count < min_count:
                min_count = count
            if count > max_count:
                max_count = count
            total += 1
            avg_count += count
    avg_count /= total
    print("Min count:", min_count)
    print("Max count:", max_count)
    print("Avg count:", avg_count)
    print("Total count:", total)
    
    print("Box length distribution:")
    for box_length, count in test_box_length_distribution.items():
        print(f"{box_length}: {count}")

    # plot histograms of gpt and qwen scores
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_gpt_scores, bins=100, alpha=0.5, label="train")
    plt.hist(test_gpt_scores, bins=100, alpha=0.5, label="test")
    plt.xlabel("GPT score")
    plt.ylabel("Count")
    plt.title("GPT score distribution")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(train_qwen_scores, bins=100, alpha=0.5, label="train")
    plt.hist(test_qwen_scores, bins=100, alpha=0.5, label="test")
    plt.xlabel("Qwen score")
    plt.ylabel("Count")
    plt.title("Qwen score distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()
    with open("final_train_annotations.json", "w") as f:
        json.dump(train_annotations, f)
    with open("final_test_annotations.json", "w") as f:
        json.dump(test_annotations, f)
    

   