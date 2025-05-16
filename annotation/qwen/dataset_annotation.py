"""
File to create the dataset class to load images after deduplication with the car boxes for annotation with VLM.
"""

import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torchvision import transforms

class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, detection_path, deduplication_path, class_name, transform=None):
        self.detection_path = detection_path
        self.deduplication_path = deduplication_path
        self.class_name = class_name
        self.transform = transform if transform else transforms.ToTensor()

        # Load image paths after deduplication
        with open(os.path.join(deduplication_path, class_name, "unique_images.txt"), "r") as f:
            self.image_paths = f.readlines()

        # Load detections (before deduplication)
        with open(os.path.join(detection_path, class_name, "detections.json"), "r") as f:
            self.detections = json.load(f)  # {image_name: [detection1, detection2, ...]}, detection: {"box": [x, y, w, h], "label": label} 

        # Flatten detections 

        self.flat_detections = []
        for image_path in self.image_paths:
            image_path = image_path.strip()
            image_name = os.path.basename(image_path)
            detections = self.detections[image_name]
            detections = [detection for detection in detections if int(detection["label"]) in [2,5,6,7]] # car, bus, truck, motorbike (based on COCO labels)
            for detection in detections:
                self.flat_detections.append((image_path, detection))
        
    def __len__(self):
        return len(self.flat_detections)
    
    def __getitem__(self, index):
        return self.flat_detections[index]
    
    
    def save_crops_batch(self, image_paths, boxes, output_dir):
        """
        Save crops for a batch of images.
        Inputs:
            image_paths: list of image paths
            detections: {"box": batch of boxes, "label": batch of labels, "score": batch of scores}
        """
        os.makedirs(output_dir, exist_ok=True)


        crop_names = []
        for i,(image_path, box) in enumerate(zip(image_paths, boxes)):
            image = Image.open(image_path).convert("RGB")
            box = convert_bbox(box)
            crop = image.crop(box)
            # resize crop to 224x224
            crop = crop.resize((224, 224))
            crop.save(os.path.join(output_dir, f"{i}.jpg"))
            crop_names.append(f"{i}.jpg")
        return crop_names
    
        crop_names = []
        for image_path, detection in zip(image_paths, detections):  
            image = Image.open(image_path).convert("RGB")
            box = detection["box"]
            box = convert_bbox(box)
            crop = image.crop(box)
            crop.save(os.path.join(output_dir, f"{i}.jpg"))
            crop_names.append(f"{i}.jpg")
        return crop_names
        """os.makedirs(output_dir, exist_ok=True)
        crop_names = []
        to_pil = transforms.ToPILImage()
        for i,(image, detection, image_path) in enumerate(zip(images, detections, image_paths)):
            box = detection["box"]
            box = convert_bbox(box)
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            _, height, width = image.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

            crop = image[:, y1:y2, x1:x2]

            crop = to_pil(crop)
            crop.save(os.path.join(output_dir, f"{i}.jpg"))
            crop_names.append(f"{i}.jpg")
        return crop_names"""


def cars_collate_fn(batch):
    """
    Custom collate function for the dataset.
    """
    image_paths = [item[0] for item in batch]
    boxes = [item[1]["box"] for item in batch]
    labels = [item[1]["label"] for item in batch]
    scores = [item[1]["score"] for item in batch]
    return image_paths, boxes, labels, scores

def convert_bbox(bbox):
    """
    Convert bbox from (x, y, w, h) to (x1, y1, x2, y2) where (x,y) are center of the box.
    """
    x, y, w, h = bbox
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_path", type=str, help="Path to detections", default=os.path.join(os.environ['HOME'], 'cars_detections'))
    parser.add_argument("--deduplication_path", type=str, help="Path to deduplicated images", default=os.path.join(os.environ['HOME'], 'cars_deduplicated'))
    parser.add_argument("--class_name", type=str, help="Class name", default="acura")
    args = parser.parse_args()

    dataset = CarsDataset(args.detection_path, args.deduplication_path, args.class_name)
    with open("coco_labels.json", "r") as f:
        coco_labels = json.load(f) # map from class id to class name
    os.makedirs("outputs/dataset_annotation", exist_ok=True)

    print(len(dataset))
    exit()
    from tqdm import tqdm
    for i in tqdm(range(50)):
        image, detections, image_path = dataset[i]
        image_name = os.path.basename(image_path)
        crops = dataset.get_crops(image, detections)

        # plot crops
        fig, axs = plt.subplots(1, min(4, len(crops)+1), figsize=(20, 10))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        for i, crop in enumerate(crops):
            if i >= 3:
                break
            axs[i+1].imshow(crop)
            axs[i+1].set_title("Crop {}".format(i))
        plt.savefig(f"outputs/dataset_annotation/{image_name}")


        """# plot image with detections
        plt.imshow(image)
        for detection in detections:
            box = detection["box"]
            box = convert_bbox(box)
            label = detection["label"]
            label = coco_labels[str(int(label))]
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='r', lw=2))
            plt.text(box[0], box[1], label, color='r')
        plt.savefig("outputs/dataset_annotation/{}.png".format(image_name))
        plt.close()

"""


        