from operator import not_
import os
import json
import argparse
from httpx import delete
from matplotlib.pyplot import cla
from tqdm import tqdm
import re
from unidecode import unidecode

brands_with_underscore = ["alfa_romeo", "aston_martin", "land_rover"]
def normalize_model_name(model_name):
    model_name = model_name.lower().strip()
    model_name = unidecode(model_name)
    model_name = model_name.replace(" ", "_")

    # remove duplicate brand names (ram_ram 2500 -> ram_2500)
    model_name = re.sub(r'\b(\w+)_\1\b', r'\1', model_name)  # "audi_audi 100" → "audi_100"

    model_name = model_name.replace(" ", "_")

    # standardize hyphens
    model_name = re.sub(r'(\w)-(\d+)', r'\1\2', model_name)  # "oldsmobile_f-85" → "oldsmobile_f85"

    parts = model_name.split("_")
    if len(parts) > 2 and parts[-1] == parts[-2]:
        parts.pop()
    model_name = "_".join(parts)

    return model_name

def class_equals(gpt_4o, qwen):
    return gpt_4o in qwen or qwen in gpt_4o

def class_in_distribution(gpt_4o, qwen_class_counts):
    for qwen_class in qwen_class_counts:
        if gpt_4o in qwen_class or qwen_class in gpt_4o:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map GPT-4o class distribution to Qwen class distribution.")
    parser.add_argument("--annotations", type=str, help="Path to annotations JSON file", default="outputs/normalized_test_annotations.json")
    parser.add_argument("--parsed_results", type=str, help="Path to parsed results JSON file", default="chatgpt/parsed_results.json")
    parser.add_argument("--output_file", type=str, help="Path to save mapped results", default="chatgpt/mapped_results.json")
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        test_annotations = json.load(f)

    with open(args.parsed_results, "r") as f:
        parsed_results = json.load(f)

    # get real distribution for qwen
    qwen_class_counts = {}
    for crop_path in parsed_results:
        qwen_class = parsed_results[crop_path]["qwen"]
        if qwen_class not in qwen_class_counts:
            qwen_class_counts[qwen_class] = 0
        qwen_class_counts[qwen_class] += 1

    mapped_results = {}
    not_in_distribution = 0
    #ignore_brands = ["smart", "toy", "ds", "rover","lego","unknown"]
    ignore_brands = ["toy", "ds", "rover","lego","not","unknown"]
    for crop_path in tqdm(parsed_results):
        crop_results = parsed_results[crop_path]
        gpt_crop = crop_results["gpt-4o"]
        gpt_4o = gpt_crop["model"]
        gpt_4o_model_probability = gpt_crop["model_probability"]
        gpt_4o_car_probability = gpt_crop["car_probability"]
        in_distribution = True
        for brand in ignore_brands:
            if gpt_4o.split("_")[0] == brand:
                in_distribution = False
                break
                
        qwen = crop_results["qwen"]
        in_distribution = (class_equals(gpt_4o, qwen) or class_in_distribution(gpt_4o, qwen_class_counts)) and in_distribution
        mapped_results[crop_path] = {"pred": gpt_4o,
                                     "model_probability": gpt_4o_model_probability,
                                     "car_probability": gpt_4o_car_probability,
                                     "qwen":crop_results["qwen"],
                                     "in_distribution": in_distribution}
        if not in_distribution:
            not_in_distribution += 1
    print(f"Total not in distribution: {not_in_distribution} ({not_in_distribution/len(parsed_results)*100:.2f}%)")

    # map underrepresented classes to unknown
    valid_classes = set()
    valid_brands = set()
    qwen_class_counts = {}
    for image_id in tqdm(test_annotations):
        image_annotations = test_annotations[image_id]
        for box in image_annotations["boxes"]:
            if box["brand"]=="unknown":
                continue
            elif box["underrepresented"]:
                box["brand"] = box["brand"].replace("-", "").replace(" ", "_")
                valid_brands.add(box["brand"])
                valid_classes.add(box["brand"] + "_unknown")
                if f"{box['brand']}_unknown" not in qwen_class_counts:
                    qwen_class_counts[f"{box['brand']}_unknown"] = 0
                qwen_class_counts[f"{box['brand']}_unknown"] += 1
            else:
                box["brand"] = box["brand"].replace("-", "").replace(" ", "_")
                box["model"] = box["model"].replace("-", "")
                class_name = box["brand"] + "_" + box["model"]
                class_name = normalize_model_name(class_name)
                valid_classes.add(class_name)
                valid_brands.add(box['brand'].replace(" ","_"))
                if class_name not in qwen_class_counts:
                    qwen_class_counts[class_name] = 0
                qwen_class_counts[class_name] += 1
  
    print(f"Total valid classes: {len(valid_classes)}")
    print(f"Total valid brands: {len(valid_brands)}")
    gpt_results = {}

    """  brands_with_underscore = ["alfa_romeo", "aston_martin", "land_rover"]

    brands_without_underscore = [brand for brand in valid_brands if brand not in brands_with_underscore]"""

    print(sorted(valid_brands))
    print(sorted(valid_classes))
    not_parsed = []
    for crop_path in tqdm(mapped_results):
        crop_results = mapped_results[crop_path]
        gpt_4o = crop_results["pred"]
        model_probability = crop_results["model_probability"]
        car_probability = crop_results["car_probability"]
        in_distribution = crop_results["in_distribution"]            
        if not in_distribution:
            continue
        if "mitsubishi_lancer_evolution" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mitsubishi_lancer_evolution", "underrepresented":False}
        elif "mazda6_wagon" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mazda_mazda6", "underrepresented":False} 
        elif "classic_mini_cooper" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mini_mini_cooper", "underrepresented":False}
        elif "classic_mini" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mini_classic", "underrepresented":False} 
        elif "mazda2" in gpt_4o or "mazda5" in crop_results:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mazda_unknown", "underrepresented":True}
        elif gpt_4o.startswith("mg"):
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"mg_unknown", "underrepresented":True}
        elif "land_rover_range_rover_sport" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"land_rover_range_rover_sport", "underrepresented":False}
        elif "land_rover_range_rover" in gpt_4o:
            gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":"land_rover_range_rover", "underrepresented":False}
        else:
            for valid_class in valid_classes:
                if class_equals(gpt_4o, valid_class):
                    gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":valid_class, "underrepresented":False} 
                    break
            else:
                if "smart" in gpt_4o:
                    not_parsed.append(gpt_4o)
                    continue    
                for brand in valid_brands:
                    gpt_parts = gpt_4o.split("_")
                    if gpt_parts[0]==brand or "_".join(gpt_parts[:2])==brand:
                        gpt_results[crop_path] = {"gpt-4o":crop_results, "model_probability":model_probability, "car_probability": car_probability, "class":f"{brand}_unknown", "underrepresented":True}
                        break
                else:
                    #print(f"Class {gpt_4o} not found")
                    not_parsed.append(gpt_4o)
    print(f"Total not parsed: {len(not_parsed)}")
    #print(sorted(not_parsed))

    # check that all classes in gpt_results are in valid_classes
    for crop_path in gpt_results:
        assert gpt_results[crop_path]["class"] in valid_classes, f"Class {gpt_results[crop_path]} not in valid classes"
    # check new class distribution
    new_class_counts = {}
    for crop_path in gpt_results:
        class_name = gpt_results[crop_path]["class"]
        if class_name not in new_class_counts:
            new_class_counts[class_name] = 0
        new_class_counts[class_name] += 1
    
    class_counts = {k: v for k, v in sorted(new_class_counts.items(), key=lambda item: item[0])}
    print(f"New class distribution:")
    print(class_counts)

    print(f"Total Qwen classes: {len(qwen_class_counts)}")
    print(f"Total GPT-4o classes: {len(class_counts)}")

    # check class that are not in qwen
    not_in_qwen = []
    for gpt4o_class in class_counts:
        for qwen_class in qwen_class_counts:
            if gpt4o_class==qwen_class:
                break
        else:
            not_in_qwen.append(gpt4o_class)
    print(f"Classes not in Qwen:")
    print(not_in_qwen)

    # check class that are not in gpt-4o (lost classes)
    lost_classes = []
    for qwen_class in qwen_class_counts:
        for gpt4o_class in class_counts:
            if gpt4o_class==qwen_class:
                break
        else:
            lost_classes.append(qwen_class)
    print(f"Lost classes:")
    print(lost_classes)


    # check number of  samples in unknown classes with gpt-4o and qwen
    unknown_gpt_4o = 0
    unknown_qwen = 0
    
    for class_name in class_counts:
        if "unknown" in class_name:
            unknown_gpt_4o += class_counts[class_name]
    
    for class_name in qwen_class_counts:
        if "unknown" in class_name:
            unknown_qwen += qwen_class_counts[class_name]
    
    print(f"Total unknown samples in GPT-4o: {unknown_gpt_4o}")
    print(f"Total unknown samples in Qwen: {unknown_qwen}")

    # 

    # check how many samples have been lost
    total_qwen = sum(qwen_class_counts.values())
    total_gpt_4o = sum(class_counts.values())
    print(f"Total samples in Qwen: {total_qwen}")
    print(f"Total samples in GPT-4o: {total_gpt_4o}")
    print(f"Total samples lost: {total_qwen - total_gpt_4o}")

    with open(args.output_file, "w") as f:
        json.dump(gpt_results, f)