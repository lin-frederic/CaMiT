import os
import json
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

import sys
path = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(path, "sam2")
if sam2_path not in sys.path:
    sys.path.append(sam2_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


sam2_checkpoint = os.path.join(sam2_path, "checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def containment_ratio(box1, mask2):
    # Calculate area of mask2 / area of box1
    # This is to check if the car in the mask2 is hidding the car in the box1 (high occlusion)

    x1,y1,x2,y2 = box1
    mask1 = np.zeros_like(mask2, dtype=np.uint8)
    mask1[y1:y2,x1:x2] = 1
    mask2 = mask2.astype(np.uint8)

    # get area of mask2 in box1
    mask2_in_box1 = mask2 * mask1 # intersection between mask2 and box1
    mask2_area = np.sum(mask2_in_box1)


    box1_area = (x2-x1)*(y2-y1)

    return mask2_area/box1_area if box1_area > 0 else 0


def intersect(box1, box2):
    cx1,cy1,w1,h1 = box1
    cx2,cy2,w2,h2 = box2
    x1_min, x1_max = cx1-w1/2, cx1+w1/2
    y1_min, y1_max = cy1-h1/2, cy1+h1/2

    x2_min, x2_max = cx2-w2/2, cx2+w2/2
    y2_min, y2_max = cy2-h2/2, cy2+h2/2

    if x1_min > x2_max or x1_max < x2_min:
        return False
    if y1_min > y2_max or y1_max < y2_min:
        return False
    return True

class OcclusionDataset(Dataset):
    def __init__(self,annotations):
        self.data = []
        for image_id, image_data in annotations.items():
            #image_path = image_data["image_path"].replace("/home/users/flin","/home/fredericlin")
            image_path = image_data["image_path"]
            boxes = image_data["boxes"]
            boxes = [box for box in boxes if "gpt_class" in box]
            # keep only images with intersecting boxes
            new_boxes = []
            has_seen = [False]*len(boxes)
            for i, box in enumerate(boxes):
                for j in range(i+1, len(boxes)):
                    if has_seen[j]:
                        continue
                    if intersect(box["box"], boxes[j]["box"]):
                        if not has_seen[i]:
                            has_seen[i] = True
                            new_boxes.append(box)
                        if not has_seen[j]:
                            new_boxes.append(boxes[j])
                            has_seen[j] = True
                        break
            
            if new_boxes:
                self.data.append((image_path, new_boxes))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, boxes = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_boxes = [[int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)] for box in boxes for cx,cy,w,h in [box["box"]]]
        return image, np.array(input_boxes), image_path

def batch_collate_fn(batch):
    images = [b[0] for b in batch]
    boxes = [b[1] for b in batch]
    image_paths = [b[2] for b in batch]
    return images, boxes, image_paths


def batch_process(predictor, dataloader, output_file):
    total = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            processed_images = json.load(f)
    else:
        processed_images = {}
    with tqdm(total=len(dataloader)) as pbar:
        for images, batch_input_boxes, image_paths in dataloader:
            total += len(image_paths)
            images = [image for image, image_path in zip(images, image_paths) if image_path not in processed_images]
            batch_input_boxes = [box for box, image_path in zip(batch_input_boxes, image_paths) if image_path not in processed_images]
            image_paths = [image_path for image_path in image_paths if image_path not in processed_images]
            if not images:
                pbar.update(1)
                continue
            predictor.set_image_batch(images)
            masks_batch, scores_batch, _ = predictor.predict_batch(
                None,
                None,
                box_batch=batch_input_boxes,
                multimask_output=True
            )

            for batch_index, image_path in enumerate(image_paths):
                input_boxes = batch_input_boxes[batch_index]
                masks = masks_batch[batch_index]
                scores = scores_batch[batch_index]

                drop_image = False
                for i, box_i in enumerate(input_boxes): # check if there is another car that is occluding the car in box_i
                    for j, mask_j in enumerate(masks):
                        if i==j:
                            continue
                        mask_j = mask_j[np.argmax(scores[j])]
                        ratio = containment_ratio(box_i, mask_j)
                        if ratio >= 0.2: #  from manual inspection, 0.2 is a good threshold
                            drop_image = True
                            break
                    if drop_image:
                        break
                processed_images[image_path] = drop_image
            # update progress bar with the filtering ratio
            filtered = sum(processed_images.values())
            with open(output_file, "w") as f:
                json.dump(processed_images, f)
            pbar.set_postfix({"Filtered": f"{filtered}/{total} ({filtered/total:.2f})"})
            pbar.update(1)

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to the train annotations", default="outputs/train_gpt_annotations_with_unknown.json") 
    parser.add_argument("--test_annotations", type=str, help="Path to the test annotations", default="outputs/test_gpt_annotations_with_unknown.json")
    parser.add_argument("--output", type=str, help="Path to the output json file", default="clean_outlier/clean_box.json")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing", default=4)
    parser.add_argument("--workers", type=int, help="Number of workers for dataloader", default=4)
    args = parser.parse_args()

    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)

    grouped_annotations = {**test_annotations, **train_annotations} # test then train
    dataset = OcclusionDataset(grouped_annotations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=batch_collate_fn)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("No GPU available")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device)
    predictor = SAM2ImagePredictor(sam2_model)
    batch_process(predictor, dataloader, args.output)


