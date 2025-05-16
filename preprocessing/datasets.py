import os
import torch
import torchvision
from torchvision import transforms
import json
from tqdm import tqdm

class DatasetFromBoxAnnotations():
    """
    Dataset from box annotations
    """
    def __init__(self, data_path, annotations_path, class_name,  transform=None):
        with open(os.path.join(annotations_path, class_name, "detections.json"), "r") as f:
            annotations = json.load(f)
        image_names = list(annotations.keys())
        image_paths = [os.path.join(data_path, class_name, x) for x in image_names]
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid_paths.append(path)
            else:
                os.remove(path)
                print(f"Removed {path} due to size 0")
        image_paths = valid_paths

        self.loader = torchvision.datasets.folder.pil_loader
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __getitem__(self, index):
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.annotations[os.path.basename(path)], path
    
    def __len__(self):
        return len(self.image_paths)


def get_transform(train=False):
    """
    Get data augmentation and normalization transforms
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

def collate_fn_annotations(batch):
    """
    Collate function for dataloader from annotations
    """
    images, annotations, paths = zip(*batch)
    images = torch.stack(images)
    return images, annotations, paths

def get_dataloader_from_annotations(data_path, annotations_path, class_name, batch_size=64, num_workers=6, shuffle=False):
    transform = get_transform(train=False)
    dataset = DatasetFromBoxAnnotations(data_path, annotations_path, class_name, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_annotations)
    return dataloader


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, metadata, transform=None):
        """
        Input: 
        data_path: Path to class images
        metadata: {image_name: {date, user_id}} (class metadata, should have constant year)
        """
        self.data_path = data_path
        self.metadata = metadata
        self.loader = torchvision.datasets.folder.pil_loader
        self.transform = transform
        self.image_paths = [os.path.join(data_path, x+".jpg") for x in metadata]
        valid_paths = []
        for path in tqdm(self.image_paths):
            try: # load image to check if it is valid
                sample = self.loader(path)
                valid_paths.append(path)
            except:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Removed {path} due to invalid image")
                else:
                    print(f"Path {path} does not exist")
            
        self.image_paths = valid_paths



    def __getitem__(self, index):
        """
        Get item from dataset.
        Be aware that image paths can be updated based on class name and time interval
        """
        path = self.image_paths[index]
        image_name = os.path.basename(path).split(".")[0]
        image_metadata = self.metadata[image_name]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, image_metadata, path
    
    def __len__(self):
        return len(self.image_paths)

def collate_fn_metadata(batch):
    """
    Collate function for dataloader from metadata
    """
    images, metadata, paths = zip(*batch)
    images = torch.stack(images)
    return images, metadata, paths

def get_dataloader_from_metadata(data_path, metadata, class_name, time_period, batch_size=64, num_workers=6, shuffle=False):
    """
    Input:
    data_path: Path to data
    metadata: {class_name: {image_name: {date, user_id}}} (global metadata)
    """

    transform = get_transform(train=False)


    class_path = os.path.join(data_path, class_name)
    class_metadata = metadata[class_name]

    time_metadata = {k: v for k, v in class_metadata.items() if v["date"] in time_period}

    dataset = TimeDataset(class_path, time_metadata, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_metadata)
    
    return dataloader


        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data", default="/scratch_global/cars_time2/images")
    parser.add_argument("--annotations_path", type=str, help="Path to annotations", default="/scratch_global/cars_time2/filtered_images")
    parser.add_argument("--metadata_path", type=str, help="Path to metadata", default="metadata.json")
    parser.add_argument("--class_names", type=str, nargs="+", help="List of class names", default=["toyota"])
    parser.add_argument("--year", type=int, help="Year to filter", default=2017)
    parser.add_argument("--mode", type=int, help="Mode to run", default=0) # 0: box annotations, 1: metadata
    args = parser.parse_args()

    data_path = args.data_path
    annotations_path = args.annotations_path
    metadata_path = args.metadata_path

    if args.mode == 0:
        dataset = DatasetFromBoxAnnotations(data_path, annotations_path, args.class_names[0])
        print(len(dataset))
        print(dataset.image_paths[:5])
        for i in range(5):
            # check path exists
            assert os.path.exists(dataset.image_paths[i]), f"Path {dataset.image_paths[i]} does not exist"
        print(dataset[0])
        print()

        dataloader = get_dataloader_from_annotations(data_path, annotations_path, args.class_names[0], batch_size=64,num_workers=1)
        print(len(dataloader))
        for images, annotations, paths in dataloader:
            print(images.shape)
            print(annotations)
            print(paths)
            break
    elif args.mode == 1:

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        class_path = os.path.join(data_path, args.class_names[0], args.class_names[0])    
        class_metadata = metadata[args.class_names[0]]
        time_metadata = {k: v for k, v in class_metadata.items() if v["date"] == args.year}

        dataset = TimeDataset(class_path, time_metadata)
        print(len(dataset))
        print(dataset.image_paths[:5])
        for i in range(5):
            # check path exists
            assert os.path.exists(dataset.image_paths[i]), f"Path {dataset.image_paths[i]} does not exist"
        print(dataset[0])
        print()

        dataloader = get_dataloader_from_metadata(data_path, metadata, args.class_names[0], args.year, batch_size=64,num_workers=1)
        print(len(dataloader))
        for images, metadata, paths in dataloader:
            print(images.shape)
            print(metadata)
            print(paths)
            break