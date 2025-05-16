"""
This script filters duplicates in a directory of images using a pretrained model.
"""
import os
import torch
from datasets import get_dataloader_from_annotations
from models import get_model, get_embeddings
from tqdm import tqdm
import json

def filter_duplicates(dataloader,model, device, threshold=0.9, dinov2=True):
    """
    Filter duplicates in a directory of images
    Might need to batchify this
    """
    embeddings, image_paths = get_embeddings(model, dataloader, device)
    num_samples = embeddings.shape[0]
    unique_indices = []
    duplicates = {} # {unique_image_path: [duplicates paths]}

    for i in tqdm(range(num_samples)):
        is_duplicate = False
        if len(unique_indices) > 0:
            # embeddings[i].unsqueeze(0): (1,D)
            # embeddings[unique_indices]: (N,D) where N is the number of unique samples
            # calculate cosine similarity between the current embedding and all previous unique embeddings (maybe batchify this)
            # right now done on cpu
            similarity = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[unique_indices])


            if sum(similarity > threshold) > 0:
                is_duplicate = True
                # embedding i is a duplicate, find the corresponding unique embeddings
                corresponding_indices = torch.where(similarity > threshold)[0]
                corresponding_image_paths = [image_paths[unique_indices[j]] for j in corresponding_indices]
                for unique_path in corresponding_image_paths:
                    duplicates[unique_path].append(image_paths[i])
                #duplicates[image_paths[i]] = [image_paths[unique_indices[j]] for j in torch.where(similarity > threshold)[0]]
        if not is_duplicate:
            unique_indices.append(i)
            duplicates[image_paths[i]] = []

    
    unique_image_paths = [image_paths[i] for i in unique_indices]
    return unique_image_paths, duplicates

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data", default=os.path.join(os.environ['HOME'], 'cars_images'))
    parser.add_argument("--annotations_path", type=str, help="Path to annotations", default=os.path.join(os.environ['HOME'], 'cars_detections'))
    parser.add_argument("--dest_path", type=str, help="Path to save images", default=os.path.join(os.environ['HOME'], 'cars_deduplicated'))
    parser.add_argument("--class_names", type=str, nargs="+", help="List of class names", default=[])
    parser.add_argument("--threshold", type=float, help="Threshold for duplicates", default=0.9)
    args = parser.parse_args()

    data_path = args.data_path
    dest_path = args.dest_path
    annotations_path = args.annotations_path

    threshold = args.threshold

    # Check if source path exists
    if not os.path.exists(data_path):
        print(f"Source path {data_path} does not exist. Exiting...")
        exit()

    # Create destination path
    os.makedirs(dest_path, exist_ok=True)
    
    if len(args.class_names) == 0:
        args.class_names = os.listdir(data_path)
        args.class_names.sort()
        
    # Check if all classes exist
    for class_name in args.class_names:
        if not os.path.exists(os.path.join(data_path, class_name)):
            print(f"Class {class_name} does not exist in {data_path}. Exiting...")
            exit()
    print("All classes exist, continuing...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Exiting...")
        exit()

    model = get_model(dinov2=True)
    model.to(device)
    model.eval()

    
    # Filter duplicates 
    for class_name in args.class_names:
        print(f"Filtering duplicates for class {class_name}...")
        os.makedirs(os.path.join(dest_path, class_name), exist_ok=True) 
        dataloader = get_dataloader_from_annotations(data_path, annotations_path, class_name, batch_size=64, num_workers=1)
        unique_image_paths,duplicates = filter_duplicates(dataloader, model, device, threshold=threshold, dinov2=True)
        # only write paths to a txt file (instead of copying images)
        with open(os.path.join(dest_path, class_name, "unique_images.txt"), "w") as f:
            f.write("\n".join(unique_image_paths))
        with open(os.path.join(dest_path, class_name, "duplicates.json"), "w") as f:
            json.dump(duplicates, f)
        print("Total number of images: ", len(dataloader.dataset))
        print(f"Number of unique images: {len(unique_image_paths)} for class {class_name}")
        print(f"Unique images for class {class_name} written to {os.path.join(dest_path, class_name, 'unique_images.txt')}")
        print()



