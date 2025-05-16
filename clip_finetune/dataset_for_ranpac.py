import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class ClassOrder(object):
    def __init__(self):
        self.class_list =  []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_list)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.class_list)}

    def __len__(self):
        return len(self.class_list)
    
    def update(self, classes):
        new_classes = set(classes) - set(self.class_list)
        self.class_list.extend(new_classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_list)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.class_list)}
    
    def class_to_index(self, cls):
        if cls not in self.class_list:
            return -1
        return self.class_to_idx[cls]
    
    def index_to_class(self, idx):
        return self.idx_to_class[idx]



class SupervisedTimeDataset(Dataset):
    def __init__(self, annotations_file, images_dir, time, transform=None):
        self.transform = transform

        time_images = os.listdir(os.path.join(images_dir, time))

        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.crops = []
        self.labels = []
        for image_name in time_images:
            image_path = os.path.join(images_dir, time, image_name)
            image_annotations = annotations[image_name.split(".")[0]]
            for box in image_annotations["boxes"]:
                if box["class"] == "unknown":
                    continue
                if box["class"] == "citroen_ds": #somehow the class is only in the test set
                    continue
                cx, cy, w, h = box["bbox"]
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                self.crops.append((image_path, (x1, y1, x2, y2)))
                self.labels.append(box["class"])
                
    def __len__(self):
        return len(self.crops)


    def __getitem__(self, idx):
        image_path, box = self.crops[idx]
        image = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = box
        image = image.crop((x1, y1, x2, y2))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, help="Path to annotations JSON file", default="cars_dataset/test_annotations.json")
    parser.add_argument("--images_dir", type=str, help="Path to images directory", default="cars_dataset/test_blurred")
    parser.add_argument("--year", type=str, help="Year to filter images", default="2007")
    args = parser.parse_args()

    with open(args.annotations_file, 'r') as f:
        annotations = json.load(f)
    all_classes = set()
    for image_name in annotations.keys():
        image_annotations = annotations[image_name]
        for box in image_annotations["boxes"]:
            if box["class"] == "unknown":
                continue
            if box["class"] == "citroen_ds":
                continue
            all_classes.add(box["class"])
    all_classes = sorted(list(all_classes))
    print("Classes:", all_classes)
    print("Number of classes:", len(all_classes))
    exit()
    # Example usage
    dataset = SupervisedTimeDataset(args.annotations_file, args.images_dir, args.year)  
    print("Number of images:", len(dataset))

    # show class distribution
    class_count = {}
    for _, label in tqdm(dataset):
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    class_count = dict(sorted(class_count.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 5))
    plt.bar(class_count.keys(), class_count.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.show()