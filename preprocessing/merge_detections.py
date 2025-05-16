import os
import shutil
import json

from tqdm import tqdm

import argparse


if __name__ == "__main__":
    """
    Merge data from the folder 2 with the folder 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data", default=os.path.join(os.environ['HOME'], 'cars_images'))
    parser.add_argument("--data_path2", type=str, help="Path to data", default=os.path.join(os.environ['HOME'], 'cars_images_2019'))

    parser.add_argument("--detections_path", type=str, help="Path to detections", default=os.path.join(os.environ['HOME'], 'cars_detections'))
    parser.add_argument("--detections_path2", type=str, help="Path to detections", default=os.path.join(os.environ['HOME'], 'cars_detections_2019'))

    args = parser.parse_args()

    data_path = args.data_path
    data_path2 = args.data_path2

    detections_path = args.detections_path
    detections_path2 = args.detections_path2

    classes = os.listdir(data_path2)
    # assert every class in data_path2 is in data_path
    for class_name in classes:
        if not os.path.exists(os.path.join(data_path, class_name)):
            print(f"Class {class_name} does not exist in {data_path}. Exiting...")
            exit()
    print("All classes exist, continuing...")

    for class_name in tqdm(classes):
        try:
            # merge detections
            with open(os.path.join(detections_path, class_name, "detections.json"), "r") as f:
                annotations = json.load(f)

            with open(os.path.join(detections_path2, class_name, "detections.json"), "r") as f:
                annotations2 = json.load(f)
        except:
            print(f"Error in class {class_name}: probably no new detections")
            continue
        
        
        annotations.update(annotations2)

        with open(os.path.join(detections_path, class_name, "detections.json"), "w") as f:
            json.dump(annotations, f)
        print(f"{class_name} detections merged")

        # move images
        images = os.listdir(os.path.join(data_path2, class_name))
        print(f"Moving {len(images)} images")
        for image in images:
            if not os.path.exists(os.path.join(data_path, class_name, image)): # only move if image does not exist
                shutil.copy(os.path.join(data_path2, class_name, image), os.path.join(data_path, class_name, image))
        print(f"{class_name} images moved")
        print()
        