import os
import json
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse


brands_to_ignore = ["lego", "hotwheels","matchbox","martini","redbull","police"]
brands_to_replace = {"corvette": "chevrolet", # corvette is a model of chevrolet
                    "maruti": "marutisuzuki", # check model to see if it is not a suzuki (maruti suzuki is a joint venture)
                    "mercedes": "mercedesbenz", # mercedes is short for mercedesbenz
                    "mercedesamg": "mercedesbenz", # mercedesamg is not a brand but a model
                    "citroën": "citroen", # replace special character to make it easy
                    "rangerover": "landrover", # rangerover is a model of landrover
                    "škoda": "skoda", # replace special character to make it easy
                    "vauxhall": "opel", # vauxhall is a subsidiary of opel
    }

model_mapping = {
        "ford": {
            "focus rs": "focus",
            "f-100": "f-series",
            "f-150": "f-series",
            "f-250": "f-series",
            "fusion": "mondeo", # fusion (us) is the same as mondeo (eu)
        },
        "chevrolet": {
            "c1": "corvette",
            "c2": "corvette",
            "c3": "corvette",
            "c/k 10": "c/k",
            "c10": "c/k"
        },
        "porsche": {
            "911 gt3 rs": "911",
            "911 gt3": "911",
            "911 turbo": "911",
        },
        "volkswagen": {
            "type 2": "transporter",
            "golf gti": "golf",
            "t1": "transporter",
            "golf r": "golf",
        },
        "audi": {
            "tt rs": "tt",
        },
        "mercedesbenz": {
            "amg": "unknown", # amg variants exist for multiple models but amg itself is not a model
            "sl": "sl-class",
        },
        "honda": {
            "civic type r": "civic"
        },
        "subaru": {
            "impreza wrx": "wrx",
            "impreza": "impreza", # wrx was the same as impreza before 2014 but now it is a separate model (maybe need to check the year before splitting)
            "impreza wrx sti": "wrx", # right now, we just group the names where wrx is present and ignore the rest of impreza models
            "wrx sti": "wrx",
        },
        "citroen": {
            "c4 picasso": "c4",
            "c4 cactus": "c4"
            # c3 r5 and c3 wrc are rally cars, not the same as c3
        },
        "ferarri": {
            "458 italia": "458",
            "458 speciale": "458",
            "458 spider": "458",
        },
        "dodge": {
            "ram 1500": "ram",
        },
        "renault": {
            "megane rs": "megane",
            "grand scenic": "scenic",
        },
        "land rover": {
            "series iii": "defender",
            "series ii": "defender",
        },
        "aston martin": {
            "v8 vantage": "vantage",
        },
        "mini": {
            "cooper": "mini cooper",
        },
        "mazda": {
            "mazda 3": "mazda3",
            "mazda 6": "mazda6",
            "miata": "mx-5",
            # keeping rx-7 and rx-8 separate for now as they are visually different
        },
        "bentley": {
            "continental gt": "continental"
        },
        "rolls-royce": {
            "phantom drophead coupé": "phantom",
        },
        "jaguar": {
            "xj6": "xj",
        },
        "cadillac": {
            "cts-v": "cts",
            "el dorado": "eldorado",
        },
        "lexus": {
            "is f": "is",
            "lx 570": "lx",
        },
        "alfa romeo": {
            "giulietta spider": "giulietta",
        },
        "opel": {
            "mana": "unknown",  # maybe opel manta but not sure
        },
        "vauxhall": "opel", # vauxhall is the same as opel
        "acura": {
            "tl": "tlx",
            "tsx": "tlx",
            "nsx": "nsx", # maybe merge into honda nsx (as it has more instances)
        },
        "mclaren":{
            "mp4-12c": "650s"
        },
        "chrysler": {
            "300c": "300",
        },
        "dacia": {
            "sandero stepway": "sandero",
        },
        "infiniti": {
            "g37": "g",
            "g35": "g",
        },
        "gmc": {
            "sierra": "silverado" # replace with chevrolet silverado
        }
    }

def replace_model(brand,model, model_mapping=model_mapping):
    brand = brand.lower().strip()
    model = model.lower().strip()

    if brand== "vauxhall":
        brand = "opel"
    elif brand == "gmc" and model == "sierra":
        brand = "chevrolet"
        model = "silverado"
    elif brand == "acura" and model == "nsx":
        brand = "honda"
    else:
        if brand in model_mapping and model in model_mapping[brand]:
            model = model_mapping[brand][model]
    model = model.replace("/","-") # replace / with - in model names (c/k -> c-k) to prevent errors in file paths
    return brand, model

def equal_boxes(box1, box2):
    # check if two boxes are equal (mind the numerical precision)
    brand1, model1 = box1["brand"], box1["model"]
    brand2, model2 = box2["brand"], box2["model"]
    if brand1 != brand2 or model1 != model2: # labels are different
        return False
    x1,y1,w1,h1 = box1["box"]
    x2,y2,w2,h2 = box2["box"]

    if abs(x1-x2) > 1 or abs(y1-y2) > 1 or abs(w1-w2) > 1 or abs(h1-h2) > 1:  # pixel precision
        return False
    return True

def filter_boxes(annotations, size_threshold=64, score_threshold=0.6):
    cleaned_annotations = {}
    for image_path in tqdm(annotations):
        image_annotations = annotations[image_path]
        boxes = image_annotations["boxes"]
        time = image_annotations["time"] 
        cleaned_image_annotations = {"time": time, "boxes": []}

        for box in boxes:
            x, y, w, h = box["box"]
            score = box["score"]
            # filter out boxes with small size or low score
            if w < size_threshold or h < size_threshold or score < score_threshold:
                continue 
            if "brand" not in box:
                box["brand"] = "unknown"
            if "model" not in box:
                box["model"] = "unknown"
            
            brand = box["brand"].lower().strip()
            model = box["model"].lower().strip()
            # replace brands and models
            if brand in brands_to_ignore:
                box["brand"] = "unknown"
            if brand in brands_to_replace:
                box["brand"] = brands_to_replace[brand]

            box["brand"], box["model"] = replace_model(box["brand"], box["model"]) 
            cleaned_image_annotations["boxes"].append(box)
        
        # only keep images with known brands
        has_known = False
        for box in cleaned_image_annotations["boxes"]:
            if box["brand"] != "unknown":
                has_known = True
                break
        if has_known:
            cleaned_annotations[image_path] = cleaned_image_annotations

    return cleaned_annotations

def deduplicate_boxes(annotations):

    # group annotations by basename
    basename_dict = {}
    for image_path in tqdm(annotations):
        basename = os.path.basename(image_path)
        if basename not in basename_dict:
            basename_dict[basename] = []
        basename_dict[basename].append(image_path)
    
    deduplicated_annotations = {}
    for basename in tqdm(basename_dict):
        # if there is only one image with the same basename, keep it
        if len(basename_dict[basename]) == 1:
            deduplicated_annotations[basename_dict[basename][0]] = annotations[basename_dict[basename][0]]
            continue
        else:
            # check if all boxes are the same
            ref_image_annotations = annotations[basename_dict[basename][0]]
            ref_boxes = ref_image_annotations["boxes"]
            same_boxes = True
            for i in range(1, len(basename_dict[basename])):
                image_annotations = annotations[basename_dict[basename][i]]
                boxes = image_annotations["boxes"]
                # check if the number of boxes is the same
                if len(boxes) != len(ref_boxes):
                    same_boxes = False
                    break
                # check if the boxes are the same
                for box1, box2 in zip(ref_boxes, boxes):
                    if not equal_boxes(box1, box2):
                        same_boxes = False
                        break
                if not same_boxes:
                    break
            
            # keep the first image if the boxes are the same
            # else drop all images (prediction is not consistent)
            if same_boxes:
                deduplicated_annotations[basename_dict[basename][0]] = ref_image_annotations
                
    return deduplicated_annotations

def keep_top_classes(annotations, min_count=2000):
    brand_models = {}
    for image_path in annotations:
        image_annotations = annotations[image_path]
        boxes = image_annotations["boxes"]
        for box in boxes:
            brand = box["brand"]
            model = box["model"]
            if brand not in brand_models:
                brand_models[brand] = {}
            if model not in brand_models[brand]:
                brand_models[brand][model] = 0
            brand_models[brand][model] += 1
    
    count_per_brand = {}
    for brand in brand_models:
        count_per_brand[brand] = sum(brand_models[brand].values())
    
    count_per_brand = {brand: count for brand, count in count_per_brand.items() if count >= min_count}
    count_per_brand.pop("unknown", None)

    count_per_model = {}
    for brand in count_per_brand:
        models = brand_models[brand]
        for model, model_count in models.items():
            if model_count >= min_count:
                if brand not in count_per_model:
                    count_per_model[brand] = {}
                count_per_model[brand][model] = model_count
    

    filtered_annotations = {}
    for image_path in annotations:
        image_annotations = annotations[image_path]
        boxes = image_annotations["boxes"]
        cleaned_image_annotations = {"time": image_annotations["time"], "boxes": []}
        has_known = False
        for box in boxes:
            brand = box["brand"]
            model = box["model"]
            box["underrepresented"] = False
            if brand not in count_per_brand:
                box["brand"] = "unknown"
                box["model"] = "unknown"
            else:
                has_known = True
                if brand not in count_per_model or model not in count_per_model[brand]:
                    #box["model"] = "unknown" # keep the model but add a field to indicate if it's an underrepresented model or not
                    box["underrepresented"] = True
            cleaned_image_annotations["boxes"].append(box)

        # only keep images with known brands
        if has_known:
            filtered_annotations[image_path] = cleaned_image_annotations

    return filtered_annotations

def clean_selected_annotations(selected_annotations_path, target_path):
    # clean boxes like in create_object_annotations but keep the original format of image:bboxes
    with open(selected_annotations_path, "r") as f:
        selected_annotations = json.load(f)
    
    cleaned_annotations = filter_boxes(selected_annotations)
    cleaned_annotations = deduplicate_boxes(cleaned_annotations)
    cleaned_annotations = keep_top_classes(cleaned_annotations)

    with open(target_path, "w") as f:
        json.dump(cleaned_annotations, f)

    return cleaned_annotations

def create_object_annotations(selected_annotations_path, object_annotations_path):
    # format: {image_name: {boxes:[boxes], time: time}}
    # for training, we want to use the boxes [{box: [cx, cy, w, h], score: score, brand: brand, model: model, image_name: image_name}]
    # ie originally image:bboxes, we want to convert to bboxes:image

    with open(selected_annotations_path, "r") as f:
        selected_annotations = json.load(f)

    object_annotations = [] # object-detection style annotations

    for image_path in tqdm(selected_annotations):
        image_annotations = selected_annotations[image_path]
        time = image_annotations["time"]
        filtered_boxes = []
        for box in image_annotations["boxes"]:  
            if box["brand"] == "unknown":
                continue
            filtered_boxes.append(box)
    
        for box in filtered_boxes:
            box["image_path"] = image_path
            box["time"] = time
            box["count"] = len(filtered_boxes) # for a box, count the number of other boxes in the same image (important for box diversity sampling)
            object_annotations.append(box)


    with open(object_annotations_path, "w") as f:
        json.dump(object_annotations, f)

    return object_annotations

def resize_keep_aspect_ratio(image, min_height, min_width):
    orig_width, orig_height = image.size

    aspect_ratio = orig_width / orig_height

    # the smaller dimension will be the minimum size so that the greater dimension will be at least the minimum size
    if orig_width >= orig_height:
        new_height = min_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = min_width
        new_height = int(new_width / aspect_ratio)

    return image.resize((new_width, new_height), Image.BILINEAR)

def resize_with_padding(image, target_size):
    """
    Resize an image to the target size with padding to keep the aspect ratio
    """

    old_width, old_height = image.size
    new_width, new_height = target_size

    ratio = min(new_width / old_width, new_height / old_height) 

    new_size = (int(old_width * ratio), int(old_height * ratio))    

    image = image.resize(new_size, Image.BILINEAR)

    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(image, ((new_width - new_size[0]) // 2, (new_height - new_size[1]) // 2))

    return new_image


def resize_transform(image):
    return resize_keep_aspect_ratio(image, 256, 256)

# Dataset class
class CropVLM(Dataset):
    def __init__(self, object_annotations_path, unzoom_factor=2, transform=None):
        with open(object_annotations_path, "r") as f:
            self.object_annotations = json.load(f)
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.brand_to_idx = None # only for supervised training, don't need for self-supervised pretraining
        self.get_all = False # get model, time, score along with crop and brand

    
    def _create_brand_to_idx_mapping(self):
        brands = set()
        brand_models = {}
        for item_annotations in self.object_annotations:
            brand = item_annotations["brand"]
            model = item_annotations["model"]
            underrepresented = item_annotations["underrepresented"]
            if brand not in brand_models:
                brand_models[brand] = []
            if underrepresented:
                if "unknown" not in brand_models[brand]:
                    brand_models[brand].append("unknown")
            elif model not in brand_models[brand]:
                brand_models[brand].append(model)
            brands.add(brand)
        brands = list(brands)
        brands.sort()
        brand_to_idx = {brand: idx for idx, brand in enumerate(sorted(brands))}

        model_idx = 0
        model_to_idx = {}
        for brand in brands:
            models = brand_models[brand]
            models.sort()
            for model in models:
                brand_model = f"{brand}_{model}"
                model_to_idx[brand_model] = model_idx
                model_idx += 1
        return brand_to_idx, model_to_idx
    
    def to_supervised(self):
        self.brand_to_idx, self.model_to_idx = self._create_brand_to_idx_mapping()

    def __len__(self):
        return len(self.object_annotations)
    
    def __getitem__(self, idx):
        item_annotations = self.object_annotations[idx]
        image_path = item_annotations["image_path"]

        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]

        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image

        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))

        # adjust the box coordinates to the cropped image
        box = [orig_x1 - unzoom_x1, orig_y1 - unzoom_y1, orig_x2 - unzoom_x1, orig_y2 - unzoom_y1]

        brand = item_annotations["brand"]
        model = item_annotations["model"]
        time = item_annotations["time"]
        score = item_annotations["score"]

        if self.transform:
            crop = self.transform(crop)

        if self.brand_to_idx:
            brand_id = self.brand_to_idx[brand] 
            #model_id = self.model_to_idx[f"{brand}_{model}"]
        #brand = self.brand_to_idx.get(brand, -1) # -1 for unknown brands

        if self.get_all:
            return crop, brand, model, time, score, box, image_path
        return crop, brand_id

class FinetuneDataset(Dataset):
    def __init__(self, annotations_path, unzoom_factor=2, transform=None, class_to_idx=None):
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f) # {image_name: {boxes:[boxes], time: time, image_path: image_path}}
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.data = []
        unique_classes = set()
        for image_name, image_annotations in self.annotations.items():
            image_path = image_annotations["image_path"]
            time = image_annotations["time"]
            for box in image_annotations["boxes"]:
                if box["brand"] == "unknown":
                    continue
                new_box = {}
                new_box["box"] = box["box"]
                if box["underrepresented"]:
                    new_box["class"] = f"{box['brand']}_unknown"
                else:
                    new_box["class"] = f"{box['brand']}_{box['model']}"
                unique_classes.add(new_box["class"])
                new_box["image_path"] = image_path
                self.data.append(new_box)
        
        if class_to_idx: # provide for test set (use the same as the training set)
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(unique_classes))}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item_annotations = self.data[idx]
        image_path = item_annotations["image_path"]

        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]

        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image

        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))

        class_name = item_annotations["class"]

        if self.transform:
            crop = self.transform(crop)

        class_id = self.class_to_idx[class_name]

        return crop, class_id

class QwenDataset(Dataset):
    def __init__(self, annotations_path, class_to_idx, unzoom_factor=2, transform=None, with_path=False):
        # class_to_idx required here (compared to FinetuneDataset)
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        with open(class_to_idx, "r") as f:
            self.class_to_idx = json.load(f)
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.data = []
        self.labels = []
        missing_count = 0
        for image_name, image_annotations in self.annotations.items():
            image_path = image_annotations["image_path"]
            time = image_annotations["time"]
            for box in image_annotations["boxes"]:
                if box["brand"] == "unknown":
                    continue
                new_box = {}
                new_box["box"] = box["box"]
                if "class" not in box:
                    missing_count += 1
                    continue
                new_box["class"] = box["class"]
                new_box["image_path"] = image_path
                self.data.append(new_box)
                self.labels.append(new_box["class"])
        print(f"Missing {missing_count} instances")

        self.with_path = with_path
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item_annotations = self.data[idx]
        image_path = item_annotations["image_path"]

        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]

        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image

        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))

        class_name = item_annotations["class"]

        if self.transform:
            crop = self.transform(crop)

        class_id = self.class_to_idx[class_name]

        if self.with_path:
            return crop, class_id, image_path

        return crop, class_id

class GPTDataset(Dataset):
    def __init__(self, annotations_path, class_to_idx, unzoom_factor=2, transform=None, with_path=False):
        # class_to_idx required here (compared to FinetuneDataset)
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        with open(class_to_idx, "r") as f:
            self.class_to_idx = json.load(f)
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.data = []
        self.labels = []
        missing_count = 0
        for image_name, image_annotations in self.annotations.items():
            image_path = image_annotations["image_path"]
            time = image_annotations["time"]
            for box in image_annotations["boxes"]:
                if box["brand"] == "unknown":
                    continue
                new_box = {}
                new_box["box"] = box["box"]
                if "gpt_class" not in box:
                    missing_count += 1
                    continue
                new_box["class"] = box["gpt_class"]
                new_box["image_path"] = image_path
                self.data.append(new_box)
                self.labels.append(new_box["class"])

        print(f"Missing {missing_count} instances")
        #class_distribution = {k: v for k, v in sorted(class_distribution.items(), key=lambda item: item[1], reverse=True)}
        #print(class_distribution)
        self.with_path = with_path


        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item_annotations = self.data[idx]
        image_path = item_annotations["image_path"]

        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]

        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image

        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))

        class_name = item_annotations["class"]

        if self.transform:
            crop = self.transform(crop)

        class_id = self.class_to_idx[class_name]

        if self.with_path:
            return crop, class_id, image_path

        return crop, class_id

class CitroenDS(Dataset):
    def __init__(self, annotations_path, unzoom_factor=2, transform=None, with_path=False):
        # class_to_idx required here (compared to FinetuneDataset)
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.data = []
        self.labels = []
        self.image_data = []
        for image_name, image_annotations in self.annotations.items():
            image_path = image_annotations["image_path"]
            time = image_annotations["time"]
            for box_idx, box in enumerate(image_annotations["boxes"]):
                if "gpt_class" not in box:
                    continue

                gpt_score = box["gpt_score"]
                qwen_score = box["qwen_score"]
                self.data.append(box)
                self.labels.append(box["gpt_class"])
                self.image_data.append({"image_name": image_name, 
                                        "image_path": image_path, 
                                        "time": time, 
                                        "box_idx": box_idx,
                                        "gpt_score": gpt_score,
                                        "qwen_score": qwen_score})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_annotations = self.data[idx]
        image_name = self.image_data[idx]["image_name"]
        image_path = self.image_data[idx]["image_path"].replace("/home/users/flin","/home/fredericlin")
        #image_path = self.image_data[idx]["image_path"]
        box_idx = self.image_data[idx]["box_idx"]
        gpt_score = self.image_data[idx]["gpt_score"]["annotation_score"][1]
        qwen_score = self.image_data[idx]["qwen_score"]["annotation_score"][1]
        time = self.image_data[idx]["time"]
        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]
        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image
        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))
        class_name = item_annotations["gpt_class"]
        if self.transform:
            crop = self.transform(crop)
        return crop, class_name, time, image_path, image_name, box_idx, gpt_score, qwen_score

                

                

class CleanedGPTDataset(Dataset):
    def __init__(self, annotations_path, unzoom_factor=2, transform=None, with_path=False):
        # class_to_idx required here (compared to FinetuneDataset)
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        self.unzoom_factor = unzoom_factor
        self.transform = transform
        self.data = []
        self.labels = []
        self.image_data = []
        for image_name, image_annotations in self.annotations.items():
            image_path = image_annotations["image_path"]
            time = image_annotations["time"]
            for box_idx, box in enumerate(image_annotations["boxes"]):
                assert "class" in box, f"Missing class in {image_name}"
                if box["class"] == "unknown":
                    continue
                gpt_score = box["gpt_score"]
                qwen_score = box["qwen_score"]
                self.data.append(box)
                self.labels.append(box["class"])
                self.image_data.append({"image_name": image_name, 
                                        "image_path": image_path, 
                                        "time": time, 
                                        "box_idx": box_idx,
                                        "gpt_score": gpt_score,
                                        "qwen_score": qwen_score})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_annotations = self.data[idx]
        image_name = self.image_data[idx]["image_name"]
        image_path = self.image_data[idx]["image_path"].replace("/home/users/flin","/home/fredericlin")
        #image_path = self.image_data[idx]["image_path"]
        box_idx = self.image_data[idx]["box_idx"]
        gpt_score = self.image_data[idx]["gpt_score"]["annotation_score"][1]
        qwen_score = self.image_data[idx]["qwen_score"]["annotation_score"][1]
        time = self.image_data[idx]["time"]
        image = Image.open(image_path).convert("RGB")

        cx, cy, w, h = item_annotations["box"]
        # coordinates of the box in the original image
        orig_x1 = cx - w / 2
        orig_x2 = cx + w / 2
        orig_y1 = cy - h / 2
        orig_y2 = cy + h / 2

        # coordinates of the box in the unzoomed image
        unzoom_w = w * self.unzoom_factor
        unzoom_h = h * self.unzoom_factor
        unzoom_x1 = max(0, cx - unzoom_w / 2)
        unzoom_x2 = min(image.width, cx + unzoom_w / 2)
        unzoom_y1 = max(0, cy - unzoom_h / 2)
        unzoom_y2 = min(image.height, cy + unzoom_h / 2)

        # crop the image
        crop = image.crop((unzoom_x1, unzoom_y1, unzoom_x2, unzoom_y2))
        class_name = item_annotations["class"]
        if self.transform:
            crop = self.transform(crop)
        return crop, class_name, time, image_path, image_name, box_idx, gpt_score, qwen_score

    
def get_crop_from_box_annotations2(source_img, box_annotations,unzoom_factor=2):
    if box_annotations["brand"] == "unknown":
        return None
    
    cx, cy, w, h = box_annotations["box"]
    x1,x2,y1,y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    # crop parameters
    unzoom_w = w * unzoom_factor
    unzoom_h = h * unzoom_factor
    unzoom_x1 = max(0, int(cx - unzoom_w / 2))
    unzoom_x2 = min(source_img.shape[1], int(cx + unzoom_w / 2))
    unzoom_y1 = max(0, int(cy - unzoom_h / 2))
    unzoom_y2 = min(source_img.shape[0], int(cy + unzoom_h / 2))

    # crop the image
    crop = source_img[unzoom_y1:unzoom_y2, unzoom_x1:unzoom_x2].copy()
    
    adjust_x1 = int(x1 - unzoom_x1)
    adjust_y1 = int(y1 - unzoom_y1)
    adjust_x2 = int(w + adjust_x1)
    adjust_y2 = int(h + adjust_y1)

    return crop, [adjust_x1, adjust_y1, adjust_x2, adjust_y2]
def get_crop_from_box_annotations(source_img, box_annotations, unzoom_factor=2, with_box = True, score_dict=None):
    if box_annotations["brand"] == "unknown":
        return None
    cx, cy, w, h = box_annotations["box"]
    x1,x2,y1,y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    # crop parameters
    unzoom_w = w * unzoom_factor
    unzoom_h = h * unzoom_factor
    unzoom_x1 = max(0, int(cx - unzoom_w / 2))
    unzoom_x2 = min(source_img.shape[1], int(cx + unzoom_w / 2))
    unzoom_y1 = max(0, int(cy - unzoom_h / 2))
    unzoom_y2 = min(source_img.shape[0], int(cy + unzoom_h / 2))

    # crop the image
    crop = source_img[unzoom_y1:unzoom_y2, unzoom_x1:unzoom_x2].copy()

    if with_box:  # Add box to the crop
        adjust_x1 = int(x1 - unzoom_x1)
        adjust_y1 = int(y1 - unzoom_y1)
        adjust_x2 = int(w + adjust_x1)
        adjust_y2 = int(h + adjust_y1)

        cv2.rectangle(crop, (adjust_x1, adjust_y1), (adjust_x2, adjust_y2), (0, 255, 0), 2)

        # Annotation label
        brand, model = box_annotations["brand"], box_annotations["model"]
        label = f"{brand}_{model}"
        
        # Adjust font properties
        font_scale = max(0.4, crop.shape[1] / 1000)  # Scale font size dynamically
        thickness = max(1,int(font_scale * 2))
        line_spacing = int(font_scale * 30)

        # Prepare legend text
        legend_text = [f"Annotation: {label}"]

        # if scores are provided, add annotation score and top-3 predictions
        if score_dict:
            annotation_score = score_dict["annotation_score"]
            top_preds = score_dict["top_preds"]

            legend_text[0] = f"Annotation: {label} ({annotation_score:.2f})"

            for rank, (pred, score) in enumerate(top_preds):
                legend_text.append(f"Rank {rank + 1}: {pred} ({score:.2f})")
        
        # Compute dynamic box size based on text size
        text_sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for text in legend_text]
        text_heights = [height for _, height in text_sizes]
        text_widths = [width for width, _ in text_sizes]

        box_width = max(text_widths) + 20
        box_height = sum(text_heights) + (len(legend_text) - 1) * line_spacing

        # Position legend at the top-left corner
        legend_x, legend_y = 5, 5

        # Draw semi-transparent overlay for legend
        overlay = crop.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + box_width, legend_y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, crop, 0.3, 0, crop)

        # Draw legend text
        text_x, text_y = legend_x + 10, legend_y + text_heights[0] + 5
        for text in legend_text:
            cv2.putText(crop, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            text_y += line_spacing
        
    return crop


def get_crops_from_image_path(image_path, selected_annotations, unzoom_factor=2, with_box=True):
    image_annotations = selected_annotations[image_path]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in image_annotations["boxes"]:
        #crop = get_crops_from_box_annotations(image, box, unzoom_factor, with_box)
        box["crop"] = get_crop_from_box_annotations(image, box, unzoom_factor, with_box)

    return image_annotations  # All boxes with their crops
    



# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_annotations", type=str, help="Path to the selected annotations", default="outputs/selected_annotations.json")
    parser.add_argument("--object_annotations", type=str, help="Path to the object annotations", default="outputs/object_annotations.json")
    parser.add_argument("--clean_selected_annotations", type=str, help="Path to the cleaned selected annotations", default="outputs/cleaned_selected_annotations.json") 
    parser.add_argument("--gpt_annotations", type=str, help="Path to the GPT annotations", default="outputs/train_gpt_annotations.json")
    parser.add_argument("--gpt_mapping", type=str, help="Path to the GPT mapping", default="outputs/gpt_class_mapping.json")
    parser.add_argument("--crop_min", type=float, help="Minimum crop size", default=0.08)
    args = parser.parse_args()

    gpt_dataset = GPTDataset(args.gpt_annotations, class_to_idx=args.gpt_mapping, unzoom_factor=2, transform=resize_transform)
    crop, class_id = gpt_dataset[0]
    import matplotlib.pyplot as plt
    plt.imshow(crop)
    plt.show()
    

    """cleaned_annotations = clean_selected_annotations(args.selected_annotations, args.clean_selected_annotations)
    print(len(cleaned_annotations))"""

    """if not os.path.exists(args.object_annotations):
        object_annotations = create_object_annotations(args.clean_selected_annotations, args.object_annotations)
        print(len(object_annotations))

    transform = transforms.Compose([
        resize_transform,
        #transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    ])
    dataset = CropVLM(args.object_annotations, transform=transform)
    dataset.to_supervised()
    dataset.get_all = True
    print(dataset.brand_to_idx)
    print(dataset.model_to_idx)

    with open(args.clean_selected_annotations, "r") as f:
        selected_annotations = json.load(f)

    os.makedirs("outputs/cropvlm", exist_ok=True)
    import random
    for i in range(10):
        random_idx = random.randint(0, len(dataset))
        crop, brand, model, time, score, box, image_path = dataset[random_idx]
        auxiliary_crops = get_crops_from_image_path(image_path, selected_annotations)
        for j, box in enumerate(auxiliary_crops["boxes"]):
            if box["brand"] == "unknown":
                continue
            crop = box["crop"]
            # save with cv2
            cv2.imwrite(f"outputs/cropvlm/crop_{i}_{j}.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        


   """