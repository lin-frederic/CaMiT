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

class TestInferenceStudentGPTCrossval(torch.utils.data.Dataset):
    def __init__(self,annotations, class_mapping, transform=None):
        with open(annotations) as f:
            self.annotations = json.load(f)
        with open(class_mapping) as f:
            self.class_mapping = json.load(f)
        self.transform = transform

        crop_paths = []

        for image_id, image_annotations in self.annotations.items():
            boxes = image_annotations["boxes"]
            for i, box in enumerate(boxes):
                if "gpt_class" not in box:
                    continue
                crop_paths.append((image_annotations["image_path"], i))

        self.crop_paths = crop_paths

    def __len__(self):
        return len(self.crop_paths)

    def __getitem__(self, idx):
        image_path, box_index = self.crop_paths[idx]
        image_id = image_path.split("/")[-1].split(".")[0]
        image = cv2.imread(image_path)

        box_annotations = self.annotations[image_id]["boxes"][box_index]
        cx, cy, w, h = box_annotations["box"]
        x1,x2,y1,y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        unzoom_w = w * 2
        unzoom_h = h * 2
        unzoom_x1 = max(0, int(cx - unzoom_w / 2))
        unzoom_x2 = min(image.shape[1], int(cx + unzoom_w / 2))
        unzoom_y1 = max(0, int(cy - unzoom_h / 2))
        unzoom_y2 = min(image.shape[0], int(cy + unzoom_h / 2))
        crop = image[unzoom_y1:unzoom_y2, unzoom_x1:unzoom_x2].copy()
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)

        if self.transform:
            crop = self.transform(crop)
        
        class_name = box_annotations["gpt_class"]
        class_id = self.class_mapping[class_name]
        return crop, class_id, class_name, image_path, box_index



    
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
    

    test_annotations = "outputs/train_gpt_annotations.json"
    gpt_subset_indices = "outputs/gpt_subset"
    class_mapping = f"outputs/gpt_class_mapping.json"
    test_dataset = TestInferenceStudentGPTCrossval(test_annotations, class_mapping, transform=transform)
    with open(class_mapping, "r") as f:
        class_mapping = json.load(f)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_mapping.keys())}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    os.makedirs("outputs/gpt_student_scores_crossval", exist_ok=True)
    for i in range(5):
        model = create_model(
        'deit_small_patch16_224',
        num_classes=214,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
        )
        checkpoint = torch.load(f'/home/users/flin/cars_time/finetune_gpt_crossval{str(i)}/best_checkpoint.pth', 
                                map_location=device,
                                weights_only=False) # needed for pytorch 2.6.0
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        model = torch.compile(model)  # Optimize execution
        with open(os.path.join(gpt_subset_indices, f"val_indices_{str(i)}.json"), "r") as f:
            test_indices = json.load(f)
        
        test_dataset_i = torch.utils.data.Subset(test_dataset, test_indices)
        #test_loader = torch.utils.data.DataLoader(test_dataset_i, batch_size=8192, shuffle=False, num_workers=8) # for 80G GPU
        test_loader = torch.utils.data.DataLoader(test_dataset_i, batch_size=4096, shuffle=False, num_workers=4) # for 40G GPU

        if not os.path.exists(f"outputs/gpt_student_scores_crossval/gpt_student_scores_{str(i)}.json"):
            results = {}
            with torch.no_grad():
                for crops, class_ids, class_names, image_paths, box_indices in tqdm(test_loader, desc=f"Running Inference {str(i+1)}/5"):
                    crops = crops.to(device)
                    logits = model(crops)
                    probs = torch.nn.functional.softmax(logits, dim=1)

                    top_probs, top_labels = torch.topk(probs, 5, dim=1)
                    top_probs = top_probs.cpu().numpy()
                    top_labels = top_labels.cpu().numpy()
                    for j, (image_path, box_index, class_id, class_name) in enumerate(zip(image_paths, box_indices, class_ids, class_names)):
                        box_index = box_index.item()
                        box_path = image_path.replace(".jpg",f"_{str(box_index)}.jpg")
                        annotation_score = float(probs[j][class_id].item())
                        top_preds =[ (idx_to_class[label], float(prob)) for label, prob in zip(top_labels[j], top_probs[j])]
                        results[box_path] = {
                            "annotation_score": (class_name, annotation_score),
                            "top_preds": top_preds
                        }

            with open(f"outputs/gpt_student_scores_crossval/gpt_student_scores_{str(i)}.json", "w") as f:
                json.dump(results, f)

            print(f"Inference done! {str(i+1)}/5")
    