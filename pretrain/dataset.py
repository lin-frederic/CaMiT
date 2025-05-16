import os
import json
import argparse
from tqdm import tqdm
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PretrainDataset2007(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        pretraining_times = ["2005-2007", "2007"]

        self.data = []

        for time in pretraining_times:
            time_dir = os.path.join(self.data_dir, time)
            time_crops = os.listdir(time_dir)
            for crop in time_crops:
                if crop.endswith(".jpg"):
                    self.data.append(os.path.join(time_dir, crop))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image    

class PretrainDataset2007WithTrain(Dataset):
    def __init__(self, pretrain_dir, train_dir, transform=None):
        self.pretrain_dir = pretrain_dir
        self.train_dir = train_dir
        self.transform = transform

        pretraining_times = ["2005-2007", "2007"]

        self.data = []

        for time in pretraining_times:
            time_dir = os.path.join(self.pretrain_dir, time)
            time_crops = os.listdir(time_dir)
            for crop in time_crops:
                if crop.endswith(".jpg"):
                    self.data.append(os.path.join(time_dir, crop))

        train_crops = os.listdir(self.train_dir)
        for crop in train_crops:
            if crop.endswith(".jpg"):
                self.data.append(os.path.join(self.train_dir, crop))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class PretrainDatasetJointWithTrain(Dataset):
    def __init__(self, pretrain_dir, train_dir, transform=None):
        self.pretrain_dir = pretrain_dir
        self.train_dir = train_dir
        self.transform = transform

        # 2005-2007,2007,2008,...,2023
        pretraining_times = ["2005-2007", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"] 
        # for joint pretraining, take 200k samples from each year
        year_count = {}
        self.data = []
        samples_per_year = 200000
        #total_samples = samples_per_year * (len(pretraining_times)-1) # exclude 2005-2007
        for year in pretraining_times:
            if year == "2005-2007":
                continue
            year_dir = os.path.join(self.train_dir, year)
            year_crops = os.listdir(year_dir)
            year_crops.sort()
            year_count[year] = 0
            for crop in year_crops:
                if crop.endswith(".jpg"):
                    if year_count[year] < samples_per_year:
                        self.data.append(os.path.join(year_dir, crop))
                        year_count[year] += 1
                    else:
                        break
            # fill with pretraining data
            year_dir = os.path.join(self.pretrain_dir, year)
            year_crops = os.listdir(year_dir)
            year_crops.sort()
            for crop in year_crops:
                if crop.endswith(".jpg"):
                    if year_count[year] < samples_per_year:
                        self.data.append(os.path.join(year_dir, crop))
                        year_count[year] += 1
                    else:
                        break
        # for joint pretraining, take everything from 2005-2007
        year_dir = os.path.join(self.pretrain_dir, "2005-2007")
        year_crops = os.listdir(year_dir)
        year_crops.sort()
        for crop in year_crops:
            if crop.endswith(".jpg"):
                self.data.append(os.path.join(year_dir, crop))
        
        print(f"Number of train images: {len(self.data)}")
        print(f"Year count: {year_count}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

class PretrainDatasetUpdate(Dataset):
    def __init__(self, pretrain_dir, train_dir, current_year, transform=None):
        self.pretrain_dir = pretrain_dir
        self.train_dir = train_dir

        year_count = {}
        self.data = []
        # Assume there are 200k samples per year
        current_year = str(current_year)    
        current_year_dir = os.path.join(self.train_dir, current_year)
        current_year_crops = os.listdir(current_year_dir)
        current_year_crops.sort()
        year_count[current_year] = 0
        for crop in current_year_crops:
            if crop.endswith(".jpg"):
                if year_count[current_year] < 200000:
                    self.data.append(os.path.join(current_year_dir, crop))
                    year_count[current_year] += 1
                else:
                    break
        current_year_dir = os.path.join(self.pretrain_dir, current_year)
        current_year_crops = os.listdir(current_year_dir)
        current_year_crops.sort()
        for crop in current_year_crops:
            if crop.endswith(".jpg"):
                if year_count[current_year] < 200000:
                    self.data.append(os.path.join(current_year_dir, crop))
                    year_count[current_year] += 1
                else:
                    break

        pretraining_times = []
        for i in range(2007, int(current_year)):
            pretraining_times.append(str(i))

        total_samples = 1950000-len(self.data) # same training size as for initial pretraining
        samples_per_year = min(200000, total_samples // len(pretraining_times))
        print(len(pretraining_times))
        print(f"Total samples: {total_samples}")    
        print(f"Samples per year: {samples_per_year}")

        for year in pretraining_times:
            year_dir = os.path.join(self.train_dir, year)
            year_crops = os.listdir(year_dir)
            year_crops.sort()
            year_count[year] = 0
            for crop in year_crops:
                if crop.endswith(".jpg"):
                    if year_count[year] < samples_per_year and len(self.data) < 1950000:
                        self.data.append(os.path.join(year_dir, crop))
                        year_count[year] += 1
                    else:
                        break
            # fill with pretraining data
            year_dir = os.path.join(self.pretrain_dir, year)
            year_crops = os.listdir(year_dir)
            year_crops.sort()
            for crop in year_crops:
                if crop.endswith(".jpg"):
                    if year_count[year] < samples_per_year and len(self.data) < 1950000:
                        self.data.append(os.path.join(year_dir, crop))
                        year_count[year] += 1
                    else:
                        break
        # fill with 2005-2007 data if needed
        if len(self.data) < total_samples:
            year_dir = os.path.join(self.pretrain_dir, "2005-2007")
            year_crops = os.listdir(year_dir)
            year_crops.sort()
            for crop in year_crops:
                if crop.endswith(".jpg"):
                    if len(self.data) < 1950000:
                        self.data.append(os.path.join(year_dir, crop))
                        year_count["2005-2007"] = year_count.get("2005-2007", 0) + 1
                    else:
                        break
        print(f"Number of train images: {len(self.data)}")
        print(f"Year count: {year_count}")
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except (OSError, IOError) as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))  # Try the next image
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--pretrain_annotations", type=str, help="Path to pretrain annotations file", default="pretrain_annotations_with_faces.json")
    parser.add_argument("--pretrain_data", type=str, help="Path to pretrain data", default="/lustre/fsn1/projects/rech/cbt/uak27eg/pretraining_car_time")
    parser.add_argument("--train_data", type=str, help="Path to train data", default="/lustre/fsn1/projects/rech/cbt/uak27eg/cars_dataset/train_crops")
    args = parser.parse_args()

    """dataset = PretrainDatasetJointWithTrain(
        pretrain_dir=args.pretrain_data,
        train_dir=args.train_data,
        transform=transforms.Compose([
            transforms.Resize(256)
        ])
    )
"""
    dataset = PretrainDatasetUpdate(
        pretrain_dir=args.pretrain_data,
        train_dir=args.train_data,
        current_year=2010,
        transform=transforms.Compose([
            transforms.Resize(256)
        ])
    )

   