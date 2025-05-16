import os
import argparse

import json

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default='insightface_results', help="path to the results directory")

    args = parser.parse_args()


    test_results = [annotation_file for annotation_file in os.listdir(args.results) if annotation_file.endswith('.json') and 'test' in annotation_file]
    train_results = [annotation_file for annotation_file in os.listdir(args.results) if annotation_file.endswith('.json') and 'train' in annotation_file]

    grouped_train_annotations = {}
    grouped_test_annotations = {}

    for annotation_file in test_results:
        with open(os.path.join(args.results, annotation_file), 'r') as f:
            annotations = json.load(f)
        for image_name, image_annotations in annotations.items():
            grouped_test_annotations[image_name] = image_annotations

    for annotation_file in train_results:
        with open(os.path.join(args.results, annotation_file), 'r') as f:
            annotations = json.load(f)
        for image_name, image_annotations in annotations.items():
            grouped_train_annotations[image_name] = image_annotations

    # Save the grouped annotations
    with open('final_test_annotations_with_faces.json', 'w') as f:
        json.dump(grouped_test_annotations, f, indent=4)

    with open('final_train_annotations_with_faces.json', 'w') as f:
        json.dump(grouped_train_annotations, f, indent=4)