import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import os
import sys
deit_path = os.path.join(os.path.dirname(__file__), "deit")
if deit_path not in sys.path:
    sys.path.append(deit_path)

from deit.datasets import build_dataset, build_transform
from deit.engine import train_one_epoch, evaluate
from deit.losses import DistillationLoss
from deit.samplers import RASampler
from deit.augment import new_data_aug_generator

import deit.models as models
import deit.models_v2 as models_v2

import deit.utils as utils

from dataset_cropvlm import resize_transform, get_crops_from_image_path, get_crop_from_box_annotations, GPTDataset, get_crop_from_box_annotations2
from torchvision import transforms

import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob

class PretrainInferenceStudent(torch.utils.data.Dataset):
    def __init__(self, images_dir, class_mapping, transform=None):
        with open(class_mapping) as f:
            self.class_mapping = json.load(f)
        self.transform = transform

        print("Scanning all image paths...")
        self.crop_paths = glob(os.path.join(images_dir, "*", "*.jpg"))  # scan all jpgs quickly
        print(f"Total images found: {len(self.crop_paths)}")

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, idx):
        box_path = self.crop_paths[idx]
        crop = Image.open(box_path).convert("RGB")
        if self.transform:
            crop = self.transform(crop)
        
        return crop, box_path

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=os.path.join(os.environ["HOME"], "pretraining_car_time"), help='Path to the images directory')
    parser.add_argument('--class_mapping', type=str, default="outputs/gpt_class_mapping.json", help='Path to the class mapping file')
    parser.add_argument("--checkpoint_path", type=str, default="/home/users/flin/cars_time/finetune_gpt/best_checkpoint.pth", help="Path to the checkpoint file")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--output_file", type=str, default="outputs/pretrain_annotations_with_gpt_student.json", help="Output file for the results") 
    args = parser.parse_args()


    # Define the transform
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        'deit_small_patch16_224',
        num_classes=214,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    model = torch.compile(model)  # Optimize execution
    class_mapping = args.class_mapping
    pretrain_dir = args.images_dir


    pretrain_dataset = PretrainInferenceStudent(pretrain_dir, class_mapping, transform=transform)
    #pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=8192, shuffle=False, num_workers=8) # for 80G GPU
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=4096, shuffle=False, num_workers=8) # for 40G GPU

    with open(class_mapping, "r") as f:
        class_mapping = json.load(f)
        
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_mapping.keys())}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    results = {}
    with torch.no_grad():
        for crops, crop_paths in tqdm(pretrain_loader, desc="Running Inference"):
            crops = crops.to(device)
            logits = model(crops)
            probs = torch.nn.functional.softmax(logits, dim=1)
            top_probs, top_labels = torch.topk(probs, 5, dim=1)
            top_probs = top_probs.cpu().numpy()
            top_labels = top_labels.cpu().numpy()
            for i, crop_path in enumerate(crop_paths):
                annotation_score = float(probs[i][0].item())
                top_preds =[ (idx_to_class[label], float(prob)) for label, prob in zip(top_labels[i], top_probs[i])]
                results[crop_path] = {
                    "annotation_score": (0, annotation_score),
                    "top_preds": top_preds
                }
    
    with open(args.output_file, "w") as f:
        json.dump(results, f)

    print("Inference done!")



        

