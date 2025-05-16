from PIL import Image
import os
import torch
from torchvision import transforms

import json


def display_memory_usage(device):
    """
    Display memory usage on a GPU
    """
    reserved_memory = torch.cuda.memory_reserved(device) # should be allocated memory (tensors) + cached memory (for future allocations)
    allocated_memory = torch.cuda.memory_allocated(device) # should be allocated memory (tensors)
    total_memory = torch.cuda.get_device_properties(device).total_memory # total memory of the GPU
    free_memory = total_memory - reserved_memory
    print(f"Total memory: {total_memory/1e9} GB")
    print(f"Allocated memory: {allocated_memory/1e9} GB")
    print(f"Reserved memory: {reserved_memory/1e9} GB")
    print(f"Free memory: {free_memory/1e9} GB")


def filter_corrupted_images(batch):
    """ Check if a batch of images is corrupted and remove corrupted images
    Input: batch (list of image paths)
    Output: batch (list of image paths without corrupted images)
    Might be long if the batch is large
    """

    new_batch = []
    for image_path in batch:
        try:
            with Image.open(image_path) as img:
                img.load()
            new_batch.append(image_path)
        # only avoid UnidentifiedImageError
        except OSError:
            print(f"Image {image_path} is corrupted")

    return new_batch

def split_per_class(classes,split,source_path):
    """
    Split classes into approximately equal subsets
    First classes of a subset have fewer images than last classes
    """
    if isinstance(classes,str):
        classes = classes.split()
    classes = [class_name.replace('"','') for class_name in classes]
    print(f"Splitting {len(classes)} classes into {split} subsets")
    # Count number of images per class
    n_images_per_class = {}
    for class_name in classes:
        try:
            n_images_per_class[class_name] = len(os.listdir(os.path.join(source_path, class_name, class_name)))
        except:
            n_images_per_class[class_name] = len(os.listdir(os.path.join(source_path, class_name, "images")))
    
    # Sort classes by number of images
    sorted_classes = sorted(n_images_per_class, key=lambda x: n_images_per_class[x])

    # Split classes into subsets
    subsets = [[] for _ in range(split)]

    for i, class_name in enumerate(sorted_classes):
        subsets[i % split].append(class_name)
    return subsets, n_images_per_class

def split_per_class_from_metadata_path(metadata_path,split):
    """
    Split classes into approximately equal subsets
    Input: metadata_path: path to metadata file
    """

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    classes = list(metadata.keys())
    print(f"Splitting {len(classes)} classes into {split} subsets")
    # Count number of images per class
    n_images_per_class = {}

    for class_name in classes:
        n_images_per_class[class_name] = len(metadata[class_name])
    
    # Sort classes by number of images
    sorted_classes = sorted(n_images_per_class, key=lambda x: n_images_per_class[x])

    # Split classes into subsets
    subsets = [[] for _ in range(split)]

    for i, class_name in enumerate(sorted_classes):
        subsets[i % split].append(class_name)
    return subsets, n_images_per_class

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split classes into subsets")
    parser.add_argument("--classes", type=str, help="List of classes", default='toyota wey suzuki porsche haval alpine tatra genesis mitsubishi volkswagen mini brilliance spyker "alfa+romeo" yutong karma tesla ford "rolls-royce" citroen ')
    parser.add_argument("--split", type=int, help="Number of subsets", default=6)
    parser.add_argument("--source_path", type=str, help="Path to data", default="/scratch_global/cars_time2/images")
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file", default="metadata.json")
    parser.add_argument("--mode", type=int, help="Mode to run", default=0) # 0: box annotations, 1: metadata
    args = parser.parse_args()
    classes = args.classes
    split = args.split
    if args.mode == 0:
        subsets, n_images_per_class = split_per_class(classes,split,args.source_path)
    elif args.mode == 1:
        subsets, n_images_per_class = split_per_class_from_metadata_path(args.metadata_path,split)
    for i, subset in enumerate(subsets):
        n_images = sum([n_images_per_class[class_name] for class_name in subset])
        # subset list > string
        subset = " ".join(subset)
        print(subset)
        print(f"Subset {i}: {n_images} images")

