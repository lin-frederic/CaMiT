import json
import os
import shutil
from fuzzywuzzy import fuzz

import matplotlib.pyplot as plt
from visualize_annotation import normalize_text, parse_annotation
from tqdm import tqdm

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image   

import logging

logging.basicConfig(level=logging.INFO)

def process_images(images, images_only_first_brand, all_images, i, count):
    for image_path in images:
        all_images.append(image_path)
        if i == 0 or count >= 1000: 
            images_only_first_brand.append(image_path)
    
def copy_images(brand):
    os.makedirs(f"outputs/filter_annotation/{brand}", exist_ok=True)
    brand_images = brand_images[brand][:10]
    print(f"Copying {len(brand_images)} images for {brand}")
    exit()
    for image_path in brand_images:
        shutil.copy(image_path, f"outputs/filter_annotation/{brand}")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, help="Path to the annotation results", default=os.path.join(os.environ['HOME'], 'cars_annotations'))
    parser.add_argument("--annotation_path2", type=str, help="Path to the annotation results", default=os.path.join(os.environ['HOME'], 'cars_model_annotations'))
    args = parser.parse_args()

    annotation_path = args.annotation_path
    annotation_path2 = args.annotation_path2
    with open("outputs/count_brands.json", "r") as f:
        count_brands = json.load(f)

    with open("outputs/brand_images.json", "r") as f:
        brand_images = json.load(f)

    count_brands = {brand: count for brand, count in count_brands.items()}
    classes = os.listdir(annotation_path)
    classes2 = os.listdir(annotation_path2)
    for i, class_name in enumerate(classes2):
        if "alfa" in class_name:
            classes2[i] = "alfa+romeo"
        elif "aston" in class_name:
            classes2[i] = "aston+martin"
        else:
            classes2[i] = class_name.split("+")[0]
    classes = list(set(classes + classes2))
    classes.sort()
    logging.info(classes)

    class_similar_brands = {}
    for class_name in tqdm(classes):
        class_name = normalize_text(class_name)
        class_similar_brands[class_name] = []
        for brand in count_brands:
            brand = normalize_text(brand)
            fuzzy_ratio = fuzz.ratio(class_name, brand)
            if fuzzy_ratio > 80:
                class_similar_brands[class_name].append((brand, count_brands[brand]))
    class_similar_brands = {class_name: sorted(brands, key=lambda x: x[1], reverse=True) for class_name, brands in class_similar_brands.items()}
    logging.info(class_similar_brands)

    images_only_first_brand = [] # images with only the first brand in class_similar_brands for every class
    all_images = [] # all images
    for class_name in tqdm(classes):
        class_name = normalize_text(class_name)
        for i, (brand, count) in enumerate(class_similar_brands[class_name]):
            # split brand_brands[brand] into 10 different threads
            images = brand_images[brand]
            batch_size = len(images) // 10
            with ThreadPoolExecutor(max_workers=10) as executor:
                for j in range(10):
                    executor.submit(process_images, images[j*batch_size:(j+1)*batch_size], images_only_first_brand, all_images, i, count)
    images_only_first_brand = list(set(images_only_first_brand)) # remove duplicates
    all_images = list(set(all_images)) # remove duplicates
    logging.info(f"Number of images with all brands: {len(images_only_first_brand)}")
    logging.info(f"Number of all images: {len(all_images)}")

    
    class_first_brands = []
    for class_name in class_similar_brands:
        if len(class_similar_brands[class_name]) > 0:
            for i, (brand, count) in enumerate(class_similar_brands[class_name]):
                if i == 0 or count >= 1000:
                    class_first_brands.append(brand)
        else:
            logging.info(f"No similar brands for {class_name}")
    logging.info([(brand, count_brands[brand]) for brand in class_first_brands])

    # check brands that have high count but are not in class_similar_brands
    count_brands = {brand: count for brand, count in count_brands.items() if count > 1000}

    brand_not_in_class = []
    for brand in tqdm(count_brands):
        if brand in ["lego", "police", "hotwheels","matchbox"]:
            logging.info(f"Not a car brand: {brand}")
            continue
        found = False
        for class_name in class_similar_brands:
            for similar_brand, _ in class_similar_brands[class_name]:
                if similar_brand == brand:
                    found = True
                    break
            if found:
                break
        if not found:
            brand_not_in_class.append(brand)
    logging.info([(brand, count_brands[brand]) for brand in brand_not_in_class])

    # check if the brand is in the annotation (ie not miss classified)
    for brand in tqdm(count_brands):
            logging.info(brand)
            images = brand_images[brand][:10]
            for i, image_path in enumerate(images):
                image_split = image_path.split("/")
                image_name = image_split[-1]
                class_name = image_split[-2]
                folder_name = image_split[-3]
                if "model" in folder_name:
                    with open(os.path.join(annotation_path2, class_name, "annotations.json"), "r") as f:
                        annotations = json.load(f)
                else:
                    with open(os.path.join(annotation_path, class_name, "annotations.json"), "r") as f:
                        annotations = json.load(f)
                image_annotation = annotations[image_name]
                image_brands = []
                for box in image_annotation:
                    image_brand, image_model = parse_annotation(box)
                    image_brands.append(image_brand)
                
                logging.info(brand in image_brands)
                if not brand in image_brands:
                    logging.info(image_annotation)
                    exit()


    exit()
    os.makedirs("outputs/filter_annotation", exist_ok=True)

    
    for brand in tqdm(count_brands):
        os.makedirs(f"outputs/filter_annotation/{brand}", exist_ok=True)
        images = brand_images[brand][:10]
        for i, image_path in enumerate(images):
            image_split = image_path.split("/")
            image_name = image_split[-1]
            class_name = image_split[-2]
            folder_name = image_split[-3]
            if "model" in folder_name:
                with open(os.path.join(annotation_path2, class_name, "annotations.json"), "r") as f:
                    annotations = json.load(f)
            else:
                with open(os.path.join(annotation_path, class_name, "annotations.json"), "r") as f:
                    annotations = json.load(f)
            image_annotation = annotations[image_name]
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)

            for box in image_annotation:
                cx, cy, w, h = box["box"]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                image_brand, image_model = parse_annotation(box)
                if image_brand == brand:
                    color = "green"
                else:
                    color = "red"
                ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, lw=2))
                image_brand, image_model = parse_annotation(box)

                text = f"{image_brand} {image_model}"
                text_x = x1
                text_y = y1 - 10

                ax.text(
                    text_x, text_y, text,
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.5, edgecolor="none", pad=2)
                    )
                
            plt.savefig(f"outputs/filter_annotation/{brand}/{image_name}")
            plt.close()

