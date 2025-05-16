import os
import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, help="Path to the annotation results", default=os.path.join(os.environ['HOME'], 'cars_annotations'))
    parser.add_argument("--annotation_path2", type=str, help="Path to the annotation results", default=os.path.join(os.environ['HOME'], 'cars_model_annotations'))  

    args = parser.parse_args()


    classes = os.listdir(args.annotation_path)
    classes2 = os.listdir(args.annotation_path2)    

    all_annotations = {}
    with tqdm(total=len(classes)+len(classes2)) as pbar:
        for folder in [classes, classes2]:
            for class_name in folder:
                selected_path = args.annotation_path if class_name in classes else args.annotation_path2
                with open(os.path.join(selected_path, class_name, "annotations.json"),"r") as f:
                    annotations = json.load(f)
                    all_annotations[class_name] = annotations
                pbar.update(1)
    with open("outputs/all_annotations.json", "w") as f:
        json.dump(all_annotations, f, indent=4)