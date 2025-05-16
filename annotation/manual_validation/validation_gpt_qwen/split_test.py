import os
import json
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mismatched_annotations", type=str, help="Path to mismatched annotations JSON file", default="mismatched_images.json")
    parser.add_argument("--num_annotators", type=int, help="Number of annotators", default=4)
    args = parser.parse_args()

    with open(args.mismatched_annotations, "r") as f:
        mismatched_annotations = json.load(f)
    
    classes = sorted(list(mismatched_annotations.keys()))
    flattened_annotations = []
    for class_name in classes:
        times = list(mismatched_annotations[class_name].keys())
        times = sorted(times, key=lambda x: int(x))
        for time in times:
            for image in mismatched_annotations[class_name][time]:
                flattened_annotations.append((class_name, time, image))
    
    subset_size = len(flattened_annotations) // args.num_annotators

    for i in range(args.num_annotators):
        start = i * subset_size
        end = start + subset_size
        if i == args.num_annotators - 1:
            end = len(flattened_annotations)
        subset = flattened_annotations[start:end]
        subset_dict = {}
        for class_name, time, image in subset:
            if class_name not in subset_dict:
                subset_dict[class_name] = {}
            if time not in subset_dict[class_name]:
                subset_dict[class_name][time] = []
            subset_dict[class_name][time].append(image)
        with open(args.mismatched_annotations.replace(".json", f"_{i}.json"), "w") as f:
            json.dump(subset_dict, f, indent=4)
        print(f"Annotator {i} has {len(subset)} images")
