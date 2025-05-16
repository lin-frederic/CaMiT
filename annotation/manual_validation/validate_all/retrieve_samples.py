import os
import json
import argparse
from tqdm import tqdm
import random
import cv2
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_student_scores", type=str, default="outputs/gpt_student_scores.json")
    parser.add_argument("--qwen_student_scores", type=str, default="outputs/qwen_student_scores.json")
    parser.add_argument("--test_annotations", type=str, default="outputs/test_gpt_annotations.json")
    parser.add_argument("--mapped_results", type=str, default="chatgpt/mapped_results.json")

    args = parser.parse_args()
    with open(args.gpt_student_scores, "r") as f:
        gpt_results = json.load(f)
    
    with open(args.qwen_student_scores, "r") as f:
        qwen_results = json.load(f)

    with open(args.test_annotations, "r") as f:
        test_annotations = json.load(f)
    
    with open(args.mapped_results, "r") as f:
        mapped_results = json.load(f)
    
    agreement = []
    total = 0
    class_distribution = {} 
    for crop_path, gpt_result in tqdm(gpt_results.items()):
        if crop_path not in qwen_results:
            continue
        qwen_result = qwen_results[crop_path]
        gpt_gt_label, gpt_gt_score = gpt_result["annotation_score"]
        qwen_gt_label, qwen_gt_score = qwen_result["annotation_score"]


        gpt_top_preds = gpt_result["top_preds"]
        gpt_labels, _ = zip(*gpt_top_preds)
        qwen_top_preds = qwen_result["top_preds"]
        qwen_labels, _ = zip(*qwen_top_preds)

        gpt_top1 = gpt_labels[0]
        qwen_top1 = qwen_labels[0]

        if gpt_gt_label == qwen_gt_label and gpt_gt_label == gpt_top1 and gpt_gt_label == qwen_top1:
            crop_dir = "/".join(crop_path.split("/")[:-1])
            crop_name = crop_path.split("/")[-1]
            image_id, box_index = crop_name.split("_") # box_index come from normalized annotations
            box_index = box_index.replace("tensor(", "").replace(").jpg", "")
            box_index = int(box_index)
            image_annotations = test_annotations[image_id]
            time = image_annotations["time"]
            if gpt_gt_label not in class_distribution:
                class_distribution[gpt_gt_label] = {}
            if time not in class_distribution[gpt_gt_label]:
                class_distribution[gpt_gt_label][time] = {}

            mapped_crop_name = f"test_crops/{image_id}_{box_index}.jpg"
            mapped_crop_result = mapped_results[mapped_crop_name]
            car_probability = mapped_crop_result["car_probability"]
            model_probability = mapped_crop_result["model_probability"]
            if car_probability < 80 or model_probability < 80: # skip low scores
                total += 1
                continue
            crop_name = f"{image_id}_{box_index}"
            class_distribution[gpt_gt_label][time][crop_name] = {
                "car_probability": car_probability,
                "teacher_score": model_probability,
                "gpt_student_score": round(gpt_gt_score*100, 2),
                "qwen_student_score": round(qwen_gt_score*100, 2),
                "image_path": image_annotations["image_path"],
                "box": image_annotations["boxes"][box_index]["box"]
            }
            agreement.append(crop_path)


        total += 1
    print(f"Agreement: {len(agreement)}/{total} ({len(agreement)/total*100:.2f}%)")
    class_distribution = {class_name: {time: crop_dict for time, crop_dict in sorted(time_dict.items())} for class_name, time_dict in class_distribution.items()}
    class_distribution = {class_name: time_dict for class_name, time_dict in sorted(class_distribution.items(), key=lambda x: sum(len(crops) for crops in x[1].values()), reverse=True)}

    # filter class with less than 5 times with more than 40 crops
    new_class_distribution = {} 
    for class_name, time_dict in class_distribution.items():
        new_time_dict = {time: crops for time, crops in time_dict.items() if len(crops) >= 40}
        if len(new_time_dict) >= 5:
            new_class_distribution[class_name] = new_time_dict
    
    class_distribution = new_class_distribution
    
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
        
        class_times = class_distribution[class_name]
        # to have time diversity, sample from all times
        class_selected_crops = {}
        times = list(class_times.keys())
        assert sum(len(crops) for crops in class_times.values()) >= N_per_class , f"Class {class_name} has less than {N_per_class} crops"
        class_count = 0
        while class_count < N_per_class:
            time = random.choice(times)
            crop_keys = list(class_times[time].keys())
            crop = random.choice(crop_keys)
            if time not in class_selected_crops:
                class_selected_crops[time] = {}
            if crop not in class_selected_crops[time]:
                class_selected_crops[time][crop] = {}
                class_count += 1
            class_selected_crops[time][crop] = class_times[time][crop]
        selected_crops[class_name] = class_selected_crops
    
    total = sum(len(crops) for time_dict in selected_crops.values() for crops in time_dict.values())

    new_selected_crops = {}

    os.makedirs("validate_all/selected_crops", exist_ok=True)
    with tqdm(total=total) as pbar: 
        for class_name, time_dict in selected_crops.items():
            os.makedirs(f"validate_all/selected_crops/{class_name}", exist_ok=True)
            new_selected_crops[class_name] = {}
            for time, crop_dict in time_dict.items():
                os.makedirs(f"validate_all/selected_crops/{class_name}/{time}", exist_ok=True)
                new_selected_crops[class_name][time] = {}
                for crop_name, crop_info in crop_dict.items():
                    image_path = crop_info["image_path"].replace("/home/users/flin","/home/fredericlin")
                    image = cv2.imread(image_path)
                    box = crop_info["box"]
                    cx, cy, w, h = box
                    x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
                    unzoom_w = w * 2
                    unzoom_h = h * 2
                    unzoom_x1 = max(0, cx-unzoom_w/2)
                    unzoom_y1 = max(0, cy-unzoom_h/2)
                    unzoom_x2 = min(image.shape[1], cx+unzoom_w/2)
                    unzoom_y2 = min(image.shape[0], cy+unzoom_h/2)

                    adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2 = x1-unzoom_x1, y1-unzoom_y1, x2-unzoom_x1, y2-unzoom_y1
                    crop = image[int(unzoom_y1):int(unzoom_y2), int(unzoom_x1):int(unzoom_x2)]
                    #cv2.rectangle(crop, (int(adjusted_x1), int(adjusted_y1)), (int(adjusted_x2), int(adjusted_y2)), (0, 255, 0), 2)
                    crop_path = f"selected_crops/{class_name}/{time}/{crop_name}.jpg"
                    new_selected_crops[class_name][time][crop_path] = crop_info
                    new_selected_crops[class_name][time][crop_path]["box"] = [int(adjusted_x1), int(adjusted_y1), int(adjusted_x2), int(adjusted_y2)]
                    cv2.imwrite(f"validate_all/{crop_path}", crop)
                    pbar.update(1)
    with open("validate_all/selected_crops.json", "w") as f:
        json.dump(new_selected_crops, f)


    
