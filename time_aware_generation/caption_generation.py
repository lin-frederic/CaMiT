import os
import json

def generate_caption_pairs_without_year(annotation_json, base_dir):
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)

    total_images = 0
    matched_annotations = 0
    valid_boxes = 0

    caption_pairs = []

    for year_folder in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year_folder)
        if not os.path.isdir(year_path):
            continue

        for image_file in os.listdir(year_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_images += 1
            image_id = os.path.splitext(image_file)[0]

            if image_id not in annotations:
                continue
            matched_annotations += 1

            record = annotations[image_id]
            boxes = record.get("boxes", [])
            if not boxes:
                continue
            valid_boxes += 1

            car_class = boxes[0].get("class", "unknown").replace("_", " ")
            caption = f"A photo of a car {car_class}"
            image_path = os.path.join(year_folder, image_file)

            caption_pairs.append({"image": image_path, "caption": caption})

    return caption_pairs, total_images, matched_annotations, valid_boxes


def generate_caption_pairs_with_year(annotation_json, base_dir):
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)

    total_images = 0
    matched_annotations = 0
    valid_boxes = 0

    caption_pairs = []

    for year_folder in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year_folder)
        if not os.path.isdir(year_path):
            continue

        for image_file in os.listdir(year_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_images += 1
            image_id = os.path.splitext(image_file)[0]

            if image_id not in annotations:
                continue
            matched_annotations += 1

            record = annotations[image_id]
            boxes = record.get("boxes", [])
            if not boxes:
                continue
            valid_boxes += 1

            car_class = boxes[0].get("class", "unknown").replace("_", " ")
            year = year_folder
            caption = f"A photo of a car {car_class} {year}"
            image_path = os.path.join(year_folder, image_file)

            caption_pairs.append({"image": image_path, "caption": caption})

    return caption_pairs, total_images, matched_annotations, valid_boxes


if __name__ == "__main__":
    annotation_file = "train_annotations.json"
    base_images_dir = "/home/data/bambaw/cars_dataset/train_blurred/"

    print("Processing without year...")
    pairs_without_year, total_imgs1, matches1, boxes1 = generate_caption_pairs_without_year(annotation_file, base_images_dir)

    print("Processing with year...")
    pairs_with_year, total_imgs2, matches2, boxes2 = generate_caption_pairs_with_year(annotation_file, base_images_dir)

    with open("caption_pairs_without_year.json", "w") as outfile1:
        json.dump(pairs_without_year, outfile1, indent=4)

    with open("caption_pairs_with_year.json", "w") as outfile2:
        json.dump(pairs_with_year, outfile2, indent=4)

    print("\n=== Summary ===")
    print(f"Total images in dataset: {total_imgs1}")
    print(f"Caption pairs (without year): {len(pairs_without_year)}")
    print(f"Caption pairs (with year): {len(pairs_with_year)}")
