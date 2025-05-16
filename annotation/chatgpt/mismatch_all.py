import os
import json
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_student_scores", type=str, default="outputs/gpt_student_scores.json")
    parser.add_argument("--qwen_student_scores", type=str, default="outputs/qwen_student_scores.json")
    parser.add_argument("--test_annotations", type=str, default="outputs/test_gpt_annotations.json")

    args = parser.parse_args()

    with open(args.gpt_student_scores, "r") as f:
        gpt_results = json.load(f)

    with open(args.qwen_student_scores, "r") as f:
        qwen_results = json.load(f)
    
    with open(args.test_annotations, "r") as f:
        test_annotations = json.load(f)

    mismatch = []
    total = 0
    matched_distribution = {} # class distribution of matched crops
    for crop_path, gpt_result in gpt_results.items():
        if crop_path not in qwen_results:
            continue
            
        # results match if all 4 models agree (gpt, qwen, gpt_student, qwen_student) 
        qwen_result = qwen_results[crop_path]
        gpt_gt_label, _ = gpt_result["annotation_score"]
        qwen_gt_label, _ = qwen_result["annotation_score"]

        gpt_top_preds = gpt_result["top_preds"]
        gpt_labels, _ = zip(*gpt_top_preds)
        qwen_top_preds = qwen_result["top_preds"]
        qwen_labels, _ = zip(*qwen_top_preds)

        gpt_top1 = gpt_labels[0]
        qwen_top1 = qwen_labels[0]

        if gpt_gt_label != qwen_gt_label:
            mismatch.append(crop_path)
        elif gpt_gt_label != gpt_top1:
            mismatch.append(crop_path)
        elif qwen_gt_label != qwen_top1:
            mismatch.append(crop_path)
        else:
            crop_name = crop_path.split("/")[-1]
            image_id, box_index = crop_name.split("_")
            box_index = box_index.replace("tensor(", "").replace(").jpg", "")
            box_index = int(box_index)
            image_annotations = test_annotations[image_id]
            time = image_annotations["time"]
            if gpt_gt_label not in matched_distribution:
                matched_distribution[gpt_gt_label] = {}
            if time not in matched_distribution[gpt_gt_label]:
                matched_distribution[gpt_gt_label][time] = []
            matched_distribution[gpt_gt_label][time].append(crop_path)

        total += 1

    print(f"Total count: {total}")
    print(f"Mismatch count: {len(mismatch)} ({len(mismatch)/total*100:.2f}%)")

    # sort by time
    matched_distribution_count = {class_name: {time: len(crops) for time, crops in time_counts.items()} for class_name, time_counts in matched_distribution.items()}
    matched_distribution_count = {class_name: {time: count for time, count in sorted(time_counts.items(), key=lambda item: int(item[0]))} for class_name, time_counts in matched_distribution_count.items()}
    matched_distribution_count = {class_name: time_counts for class_name, time_counts in sorted(matched_distribution_count.items(), key=lambda item: sum(item[1].values()), reverse=True)}

    # check average count per class-time
    avg_count = 0
    total = 0
    total_filtered = 0
    valid_classes = set()
    for class_name, time_counts in matched_distribution_count.items():
        total += len(time_counts)
        valid_time = {time: count for time, count in time_counts.items() if count >= 40}
        if len(valid_time) < 5:
            continue
        for time, count in valid_time.items():
            avg_count += count
            total_filtered += 1
            valid_classes.add(class_name)
    avg_count /= total_filtered
    print(f"Average count per class-time: {avg_count:.2f}")
    print(f"Total filtered count: {total_filtered}/{total} ({total_filtered/total*100:.2f}%)")
    print(f"Valid classes: {len(valid_classes)} / {len(matched_distribution_count)} ({len(valid_classes)/len(matched_distribution_count)*100:.2f}%)")

    # check class-time count for non valid classes

    for class_name, time_counts in matched_distribution_count.items():
        if class_name in valid_classes:
            continue
        print(f"{class_name}: {sum(time_counts.values())}")
        for time, count in time_counts.items():
            print(f"\t{time}: {count}")
        print()
    # sample N = 3000 crops from valid classes to validate the automatic selection
    # sample from 20 pre-selected classes
    import random
    N = 3000

    # sorted
    selected_classes = ['audi_a4', 'bmw_3_series', 'chevrolet_corvette', 'dacia_sandero', 'dodge_challenger',
                        'ferrari_458_italia', 'ford_fseries', 'ford_mustang', 'honda_civic', 'lamborghini_aventador',
                        'mercedesbenz_cclass', 'nissan_gtr', 'opel_corsa', 'peugeot_208', 'peugeot_308',
                        'porsche_911', 'renault_clio', 'renault_megane', 'skoda_octavia', 'toyota_corolla']


    N_per_class = N // len(selected_classes)
    print(N_per_class)
    selected_crops = {}
    for class_name in selected_classes:
        class_times = matched_distribution[class_name]
        # to have time diversity, sample from all times
        class_selected_crops = set()
        times = list(class_times.keys())
        while len(class_selected_crops) < N_per_class:
            time = random.choice(times)
            crops = class_times[time]
            if len(crops) == 0:
                continue
            crop = random.choice(crops)
            class_selected_crops.add(crop) # if crop is already selected, it will be ignored
        selected_crops[class_name] = class_selected_crops


            
    
