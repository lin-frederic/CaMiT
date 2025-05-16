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

from dataset_cropvlm import resize_transform, get_crops_from_image_path, get_crop_from_box_annotations, FinetuneDataset, get_crop_from_box_annotations2
from torchvision import transforms

import cv2
from PIL import Image
from tqdm import tqdm

class TestFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, test_path, class_to_idx, transform=None):
        self.test_path = test_path
        with open('outputs/cleaned_selected_annotations.json') as f:
            self.annotations = json.load(f)
        with open('outputs/cleaned_test_annotations.json') as f:
            self.test_annotations = json.load(f)
        
        self.transform = transform

        self.crop_paths_in_folder = []
        self.crop_paths = []

        classes = os.listdir(test_path)
        for class_name in classes:
            years = os.listdir(os.path.join(test_path, class_name))
            for year in years:
                crops = os.listdir(os.path.join(test_path, class_name, year))
                for crop in crops:
                    if crop.startswith("model"):
                        image_idx, crop_idx = crop.split("_")[1], crop.split("_")[2]
                    else:
                        image_idx, crop_idx = crop.split("_")[0], crop.split("_")[1]
                    img_path = self.test_annotations[image_idx]["image_path"]
                    self.crop_paths_in_folder.append(os.path.join(test_path, class_name, year, crop))
                    self.crop_paths.append(img_path)
        
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.crop_paths_in_folder)

    def __getitem__(self, idx):
        crop_path_in_folder = self.crop_paths_in_folder[idx]
        crop_path = self.crop_paths[idx]

        crop_name = crop_path_in_folder.split("/")[-1]
        if crop_name.startswith("model"):
            image_idx, crop_idx = crop_name.split("_")[1], crop_name.split("_")[2]
        else:
            image_idx, crop_idx = crop_name.split("_")[0], crop_name.split("_")[1]
        
        image_annotations = self.annotations[crop_path]
        image = cv2.imread(crop_path)
        box_annotations = image_annotations["boxes"][int(crop_idx)]
        crop = get_crop_from_box_annotations(image, box_annotations, with_box=False)
        # cv2 to pil
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)

        if self.transform:
            crop = self.transform(crop)
        return crop, crop_path_in_folder

    def get_crop_with_box_and_score(self, idx, score_dict):
        crop_path_in_folder = self.crop_paths_in_folder[idx]
        crop_path = self.crop_paths[idx]
        crop_name = crop_path_in_folder.split("/")[-1]
        if crop_name.startswith("model"):
            image_idx, crop_idx = crop_name.split("_")[1], crop_name.split("_")[2]
        else:
            image_idx, crop_idx = crop_name.split("_")[0], crop_name.split("_")[1]

        image_annotations = self.annotations[crop_path]
        image_path = self.test_annotations[image_idx]["image_path"]
        image = cv2.imread(image_path)
        box_annotations = image_annotations["boxes"][int(crop_idx)]
        crop, adjusted_box = get_crop_from_box_annotations2(image, box_annotations)
        score = score_dict[crop_path_in_folder]
        return crop, adjusted_box, score, crop_path_in_folder

    def visualize_with_score(self, idx, score):
        crop_path_in_folder = self.crop_paths_in_folder[idx]
        crop_path = self.crop_paths[idx]
        crop_name = crop_path_in_folder.split("/")[-1]
        if crop_name.startswith("model"):
            image_idx, crop_idx = crop_name.split("_")[1], crop_name.split("_")[2]
        else:
            image_idx, crop_idx = crop_name.split("_")[0], crop_name.split("_")[1]

        image_annotations = self.annotations[crop_path]
        image_path = self.test_annotations[image_idx]["image_path"]
        image = cv2.imread(image_path)
        box_annotations = image_annotations["boxes"][int(crop_idx)]
        crop = get_crop_from_box_annotations(image, box_annotations,score_dict=score)

        return crop

        


if __name__ == '__main__':
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

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        'deit_small_patch16_224',
        num_classes=272,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
    )

    checkpoint = torch.load('/home/users/flin/cars_time/finetune_logits/best_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    model = torch.compile(model)  # Optimize execution
    if not os.path.exists("outputs/class_to_idx.json"):
        train_dataset = FinetuneDataset("outputs/cleaned_train_annotations.json", transform=transform)
        with open("outputs/class_to_idx.json", "w") as f:
            json.dump(train_dataset.class_to_idx, f)
    
    with open("outputs/class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)

    test_dataset = TestFinetuneDataset("outputs/test_models", class_to_idx, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=4)

    if not os.path.exists("outputs/annotation_scores.json"):    
        results = {}


        with torch.no_grad():
            for crops, crop_paths in tqdm(test_loader, desc="Running Inference"):
                crops = crops.to(device)
                logits = model(crops)
                probs = torch.nn.functional.softmax(logits, dim=1)

                top_probs, top_labels = torch.topk(probs, 5, dim=1)  # Get top 5 predictions
                top_probs = top_probs.cpu().numpy()
                top_labels = top_labels.cpu().numpy()

                for i, crop_path in enumerate(crop_paths):
                    gt_label = crop_path.split("/")[-3]
                    gt_label_idx = class_to_idx[gt_label]

                    annotation_score = float(probs[i][gt_label_idx].item())
                    top_preds = [(test_dataset.idx_to_class[label], float(prob)) for label, prob in zip(top_labels[i], top_probs[i])]
                    results[crop_path] = {
                        "annotation_score": annotation_score,
                        "top_preds": top_preds
                    }
        
        with open("outputs/annotation_scores.json", "w") as f:
            json.dump(results, f)

        print("Inference done!")

    with open("outputs/annotation_scores.json", "r") as f:
        results = json.load(f)

    # calculate top1,2,3,4,5 accuracy
    for i in range(1, 6):
        correct = 0
        for crop_path, result in results.items():
            gt_label = crop_path.split("/")[-3]
            gt_label_idx = class_to_idx[gt_label]
            top_preds = [label for label, _ in result["top_preds"][:i]]
            if gt_label in top_preds:
                correct += 1
        print(f"Top {i} accuracy: {correct / len(results)}")
    

    

