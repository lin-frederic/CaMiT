import numpy as np
import torch
from torchvision import transforms
import torchvision.models as torchvision_models

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import argparse
import json
from dataset_cropvlm import CleanedGPTDataset, resize_transform



import sys
import os
import cv2
from functools import partial

moco_path = os.path.join(os.path.dirname(__file__), "moco-v3")
if moco_path not in sys.path:
    sys.path.append(moco_path)
    sys.path.append(os.path.join(moco_path, "moco"))

deit_path = os.path.join(os.path.dirname(__file__), "deit")
if deit_path not in sys.path:
    sys.path.append(deit_path)
from timm.models import create_model
import deit.models as models
import deit.models_v2 as models_v2


import moco.builder
import vits





from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cleaned_annotations", type=str, help="Path to the cleaned annotations", default="validate_score/cleaned_annotations.json")
parser.add_argument("--crop_min", type=float, help="Minimum crop size", default=0.08)
parser.add_argument("--N", type=int, help="Number of images to sample per brand and time", default=30)
parser.add_argument("--batch_size", type=int, help="Batch size", default=1024)
parser.add_argument("--model", type=str, help="Model to use", default="moco")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')


# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

parser.add_argument('--debug', action='store_true', help='debug mode') # default: False
parser.add_argument('--job', type=int, help='job number', default=0) # default: 
parser.add_argument('--split', type=int, help='Number of parallel jobs', default=8) # default: 8


args = parser.parse_args()

# Define the transform
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    resize_transform,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

visualize_transform = transforms.Compose([
    resize_transform
]) # No center crop to visualize the whole car


# Load the dataset
dataset = CleanedGPTDataset(args.cleaned_annotations, transform=transform)

# print the number of images in the dataset
print(f"Number of images in the dataset: {len(dataset)}")
# print number of classes in the dataset, self.labels is the list of classes for all instances, to have unique classes, we need to use set
print(f"Number of classes in the dataset: {len(set(dataset.labels))}")
# Load Moco model
model = moco.builder.MoCo_ViT(
        partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
        args.moco_dim, args.moco_mlp_dim, args.moco_t)

#checkpoint = torch.load('/home/users/flin/cars_time/checkpoints/checkpoint_0299.pth.tar')
checkpoint = torch.load("checkpoints/checkpoint_0299.pth.tar")
checkpoint = checkpoint['state_dict']
# remove prefixes
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)
model.eval()
# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.compile(model)

# group instances by class
class_to_image_indices = {}

for i in range(len(dataset)):
    label = dataset.labels[i]
    if label not in class_to_image_indices:
        class_to_image_indices[label] = []
    class_to_image_indices[label].append(i)
# Split the dataset into N jobs
current_job = args.job
num_jobs = args.split

all_labels = list(class_to_image_indices.keys())
classes_per_job = len(all_labels) // num_jobs
remaining_classes = len(all_labels) % num_jobs
start_index = current_job * classes_per_job + min(current_job, remaining_classes)
end_index = start_index + classes_per_job + (1 if current_job < remaining_classes else 0)

job_labels = all_labels[start_index:end_index]
test_images = {}

for label in tqdm(job_labels):
    indices = class_to_image_indices[label]
    # make subset
    dataset_subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset_subset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    image_data = {}
    with torch.no_grad():
        for instances, class_names, times, image_paths, image_names, box_indices, gpt_scores, qwen_scores in tqdm(dataloader):
            images = instances.to(device)
            output = model.predictor(model.base_encoder(images))
            for i in range(len(instances)):
                class_name = class_names[i]
                assert class_name == label, f"Class name {class_name} does not match label {label}"
                time = times[i]
                image_name = image_names[i]
                box_index = box_indices[i]
                box_name = f"{image_name}+{box_index}"
                if time not in image_data:
                    image_data[time] = {}
                image_data[time][box_name] = [output[i].cpu().numpy(),gpt_scores[i].cpu().numpy(),qwen_scores[i].cpu().numpy()]

    # kmeans clustering on the features
    n_clusters = 50

    all_box_names = []
    all_features = []
    all_gpt_scores = []
    all_qwen_scores = []
    for time, boxes in image_data.items():
        for box_name, (feature, gpt_score, qwen_score) in boxes.items():
            all_box_names.append((box_name,time))
            all_features.append(feature)
            all_gpt_scores.append(gpt_score)
            all_qwen_scores.append(qwen_score)

    all_features = np.stack(all_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_features)

    # Group boxes by time and cluster
    time_cluster_to_boxes = {}
    for i in range(len(all_box_names)):
        box_name, time = all_box_names[i]
        cluster = cluster_labels[i]
        gpt_score = all_gpt_scores[i]
        qwen_score = all_qwen_scores[i]
        if time not in time_cluster_to_boxes:
            time_cluster_to_boxes[time] = {}
        if cluster not in time_cluster_to_boxes[time]:
            time_cluster_to_boxes[time][cluster] = []
        time_cluster_to_boxes[time][cluster].append((box_name, gpt_score, qwen_score))
    
    # Sample N boxes per time using high-score priority
    time_to_sampled_boxes = {}
    for time, cluster_to_boxes in time_cluster_to_boxes.items():
        time_cluster_with_high_score = {}
        #time_cluster_with_remaining_boxes = {}
        # split high and low score samples per cluster
        for cluster, boxes in cluster_to_boxes.items():
            for box_name, gpt_score, qwen_score in boxes:
                if gpt_score > 0.8 and qwen_score > 0.8:
                    if cluster not in time_cluster_with_high_score:
                        time_cluster_with_high_score[cluster] = []
                    time_cluster_with_high_score[cluster].append((box_name, gpt_score, qwen_score))

        # count total high score samples
        count_high_score = 0
        for cluster in time_cluster_with_high_score:
            count_high_score += len(time_cluster_with_high_score[cluster])
        
        if count_high_score <= args.N:
            print(f"[{label} | {time}] Only found {count_high_score} high score samples.")
        
        candidates = []
        clusters = [c for c in time_cluster_with_high_score]
        while len(candidates) < args.N and len(clusters) > 0:
            cluster = np.random.choice(clusters)
            np.random.shuffle(time_cluster_with_high_score[cluster])
            box_name, gpt_score, qwen_score = time_cluster_with_high_score[cluster].pop()
            candidates.append(box_name)
            if len(time_cluster_with_high_score[cluster]) == 0:
                clusters.remove(cluster)
                
        time_to_sampled_boxes[time] = candidates
    print([f"{label} | {time}: {len(boxes)}" for time, boxes in time_to_sampled_boxes.items()])
    # add to test images
    for time, boxes in time_to_sampled_boxes.items():
        if time not in test_images:
            test_images[time] = set()
        for box_name in boxes:
            image_name = box_name.split("+")[0]
            test_images[time].add(image_name)

test_images = {time: list(images) for time, images in test_images.items()}
os.makedirs("test_images", exist_ok=True)
# Save the test images
with open(f"test_images/test_images_{args.job}.json", "w") as f:
    json.dump(test_images, f)


            


    