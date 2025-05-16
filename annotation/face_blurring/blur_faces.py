import os
import json
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt4
from multiprocessing import Pool

def process_image(args):
    image_id, image_data, image_dir, output_dir = args
    time = image_data['time']
    os.makedirs(os.path.join(output_dir, time), exist_ok=True)
    image_path = os.path.join(image_dir, time, f"{image_id}.jpg")
    blurred_image_path = os.path.join(output_dir, time, f"{image_id}.jpg")

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        for face in image_data['faces']:
            x1, y1, x2, y2 = face['bbox']
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            if x2 <= x1 or y2 <= y1:
                print(f"Invalid face bounding box: {face['bbox']}")
                continue

            face_region = image[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            image[y1:y2, x1:x2] = blurred_face

        cv2.imwrite(blurred_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Path to split", default="test")
    parser.add_argument("--workers", type=int, help="Number of workers for parallel processing", default=8)
    args = parser.parse_args()

    split = args.split
    assert split in ["train", "test"], "split must be either train or test"

    if split == "train":
        image_dir = "cars_dataset/train"
        with open("cars_dataset/train_annotations.json", "r") as f:
            annotations = json.load(f)
    else:
        image_dir = "cars_dataset/test"
        with open("cars_dataset/test_annotations.json", "r") as f:
            annotations = json.load(f)

    output_dir = f"cars_dataset/{split}_blurred"
    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (image_id, image_data, image_dir, output_dir)
        for image_id, image_data in annotations.items()
    ]

    # Multiprocessing
    mp.set_start_method('fork')  # or 'spawn' if 'fork' causes issues
    with Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_image, args_list), total=len(args_list)))
