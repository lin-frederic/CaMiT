import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_annotations", type=str, help="Path to the annotation results", default="outputs/all_annotations.json")  
    parser.add_argument("--count_brands", type=str, help="Path to the count brands", default="outputs/count_brands.json")
    parser.add_argument("--brand_images", type=str, help="Path to the brand images", default="outputs/brand_images.json")
    parser.add_argument("--all_metadata", type=str, help="Path to the all metadata", default="outputs/all_metadata.json")
    args = parser.parse_args()

    print("Loading input files")
    start = time.time()
    with open(args.all_annotations, "r") as f:
        all_annotations = json.load(f)

    with open(args.count_brands, "r") as f:
        count_brands = json.load(f)
    
    with open(args.brand_images, "r") as f:
        brand_images = json.load(f)
    
    with open(args.all_metadata, "r") as f:
        time_annotations = json.load(f)
    
    print(f"Loaded everything in {time.time()-start}")  

    print("Filtering brands")
    start = time.time()
    count_brands = {brand: count for brand, count in count_brands.items() if count > 1000}

    brands_to_ignore = ["lego", "hotwheels","matchbox","martini","redbull","police"]
    count_brands = {brand: count for brand, count in count_brands.items() if brand not in brands_to_ignore}

    brands_to_replace = {"corvette": "chevrolet", # corvette is a model of chevrolet
                          "maruti": "marutisuzuki", # check model to see if it is not a suzuki (maruti suzuki is a joint venture)
                          "mercedes": "mercedesbenz", # mercedes is short for mercedesbenz
                          "mercedesamg": "mercedesbenz", # mercedesamg is not a brand but a model
                          "citroën": "citroen", # replace special character to make it easy
                          "rangerover": "landrover", # rangerover is a model of landrover
                          "škoda": "skoda", # replace special character to make it easy
    }

    brand_images = {brand: brand_images[brand] for brand in count_brands.keys()}

    for brand in brands_to_ignore:
        brand_images.pop(brand, None)
        count_brands.pop(brand, None)

    for brand, new_brand in brands_to_replace.items():
        brand_images[new_brand] = brand_images.get(new_brand, []) + brand_images.get(brand, [])
        brand_images.pop(brand, None)

        count_brands[new_brand] = count_brands.get(new_brand, 0) + count_brands.get(brand, 0)
        count_brands.pop(brand, None)
    print(f"Filtered brands in {time.time()-start}")

    images = []
    for brand in brand_images:
        images.extend(brand_images[brand])
    images = list(set(images)) # remove duplicates

    selected_annotations = {} # flattened annotations
    for image_path in tqdm(images):
        split_path = image_path.split("/")
        image_name = split_path[-1]
        class_name = split_path[-2]
        if image_name not in all_annotations[class_name] or image_name.split(".")[0] not in time_annotations[class_name]:
            continue
        box_annotations = all_annotations[class_name][image_name]
        time_annotation = time_annotations[class_name][image_name.split(".")[0]]    
        selected_annotations[image_path] = {"boxes": box_annotations, "time": time_annotation}

    with open("outputs/selected_annotations.json", "w") as f:
        json.dump(selected_annotations, f, indent=4)
    
    with open("outputs/selected_brand_images.json", "w") as f:
        json.dump(brand_images, f, indent=4)