import os
import orjson
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

test_path = 'outputs/test_models'
classes = [c.name for c in os.scandir(test_path) if c.is_dir()]
print("Classes:", classes)

sample_dict = {}
box_dict = {}

for class_name in classes:
    class_path = os.path.join(test_path, class_name)
    for time_dir in os.scandir(class_path):
        if not time_dir.is_dir():
            continue
        for sample in os.scandir(time_dir.path):
            sample_split = sample.name.split('.')[0].split('_')
            source_class = sample_split[-1]
            
            if sample.name.startswith("model"):
                image_name = sample_split[1]
                box_id = sample_split[2]
            else:
                image_name = sample_split[0]
                box_id = sample_split[1]
            
            box_name = f"{image_name}_{box_id}"
            sample_dict.setdefault(image_name, []).append(box_name)
            box_dict.setdefault(box_name, []).append(sample.path)

# Remove duplicate boxes
for box_name, paths in tqdm(box_dict.items(), desc="Removing duplicates"):
    for duplicate in paths[1:]:
        os.remove(duplicate)

# Count statistics
num_images = len(sample_dict)
num_boxes = sum(len(b) for b in sample_dict.values())
num_duplicates = num_boxes - len(set(box for boxes in sample_dict.values() for box in boxes))
print(f"Images: {num_images}, Boxes: {num_boxes}, Duplicates: {num_duplicates}, Unique Boxes: {num_boxes - num_duplicates}")

# Load annotations efficiently
with open('outputs/cleaned_selected_annotations.json', 'rb') as f:
    annotations = orjson.loads(f.read())

# Convert annotations keys
#annotations = {os.path.basename(k).split('.')[0]: v for k, v in annotations.items()}
train_annotations = {}
test_annotations = {}

for image_path, image_annotations in annotations.items():
    image_name = os.path.basename(image_path).split('.')[0]
    if image_name in sample_dict:
        test_annotations[image_name] = image_annotations
        test_annotations[image_name]["image_path"] = image_path
    else:
        train_annotations[image_name] = image_annotations
        train_annotations[image_name]["image_path"] = image_path

print(f"Train Images: {len(train_annotations)}, Test Images: {len(test_annotations)}")

# Count known boxes
num_train_boxes = sum(1 for v in train_annotations.values() for b in v["boxes"] if b["brand"] != "unknown")
num_test_boxes = sum(1 for v in test_annotations.values() for b in v["boxes"] if b["brand"] != "unknown")
print(f"Train Boxes: {num_train_boxes}, Test Boxes: {num_test_boxes}")

# Check class-time distribution
valid_brand = set()
valid_class_times = set()
for class_name in classes:
    class_path = os.path.join(test_path, class_name)
    brand = class_name.split('_')[0]
    valid_brand.add(brand)
    for time_dir in os.scandir(class_path):
        if not time_dir.is_dir():
            continue
        valid_class_times.add(f"{class_name}_{time_dir.name}")

class_time_counts = {}
for image_name in sample_dict:
    image_annotations = test_annotations[image_name]
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    for box in boxes:
        class_time = f"{box['brand']}_{box['model']}_{time}"
        if class_time not in valid_class_times:
            if box['brand'] in valid_brand:
                class_time = f"{box['brand']}_unknown_{time}"
            else:
                continue
        class_time_counts[class_time] = class_time_counts.get(class_time, 0) + 1

class_time_counts = dict(sorted(class_time_counts.items(), key=lambda x: x[1], reverse=True))
print(f"Class-time counts: {class_time_counts}")
noisy_class_times = {k: v for k, v in class_time_counts.items() if v < 40}
print(f"Noisy class_times: {noisy_class_times}")

# remove noisy class_times from annotations
new_test_annotations = {}
new_train_annotations = {}

for image_name, image_annotations in tqdm(test_annotations.items(), desc="Filtering test annotations"):
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    has_noisy = False
    for box in boxes:
        if box["underrepresented"]:
            class_time = f"{box['brand']}_unknown_{time}"
        else:
            class_time = f"{box['brand']}_{box['model']}_{time}"
        
        if class_time in noisy_class_times:
            has_noisy = True
            break

    if not has_noisy: # remove all images that contain noisy class_times in the boxes
        new_test_annotations[image_name] = image_annotations

    else: # remove it from folder
        for box in sample_dict[image_name]:
            assert len(box_dict[box]) == 1 # should have been deduplicated
            os.remove(box_dict[box][0])


for image_name, image_annotations in tqdm(train_annotations.items(), desc="Filtering train annotations"):
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    has_noisy = False
    for box in boxes:
        if box["underrepresented"]:
            class_time = f"{box['brand']}_unknown_{time}"
        else:
            class_time = f"{box['brand']}_{box['model']}_{time}"
        if class_time in noisy_class_times:
            has_noisy = True
            break
    if not has_noisy: # remove all images that contain noisy class_times in the boxes
        new_train_annotations[image_name] = image_annotations

# drop all class_times in train that are not in test
test_class_times = set()
for image_name, image_annotations in new_test_annotations.items():
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    for box in boxes:
        if box["brand"] == "unknown":
            continue
        if box["underrepresented"]:
            class_time = f"{box['brand']}_unknown_{time}"
        else:
            class_time = f"{box['brand']}_{box['model']}_{time}"
        test_class_times.add(class_time)

new_train_annotations2 = {}
for image_name, image_annotations in new_train_annotations.items():
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    image_path = image_annotations["image_path"]
    has_known = False
    new_boxes = []
    for box in boxes:
        new_box = box.copy()
        if new_box["underrepresented"]:
            class_time = f"{new_box['brand']}_unknown_{time}"
        else:
            class_time = f"{new_box['brand']}_{new_box['model']}_{time}"
        if class_time not in test_class_times:
            new_box["brand"] = "unknown"
            new_box["model"] = "unknown"
            new_box["underrepresented"] = False
        new_boxes.append(new_box)
        has_known = has_known or new_box["brand"] != "unknown"

    if has_known:
        new_train_annotations2[image_name] = {"boxes": new_boxes, "time": time, "image_path": image_path}

train_class_times = set()
for image_name, image_annotations in new_train_annotations2.items():
    boxes = image_annotations["boxes"]
    time = image_annotations["time"]
    for box in boxes:
        if box["brand"] == "unknown":
            continue
        if box["underrepresented"]:
            class_time = f"{box['brand']}_unknown_{time}"
        else:
            class_time = f"{box['brand']}_{box['model']}_{time}"
        train_class_times.add(class_time)

print("Test not in Train:", [c for c in test_class_times if c not in train_class_times])
print("Train not in Test:", [c for c in train_class_times if c not in test_class_times])
    
print(f"Train Images: {len(new_train_annotations)}, Test Images: {len(new_test_annotations)}")
print(f"Train Images2: {len(new_train_annotations2)}")

new_num_train_boxes = 0
new_num_test_boxes = 0
for v in new_train_annotations.values():
    for b in v["boxes"]:
        if b["brand"] != "unknown":
            new_num_train_boxes += 1
for v in new_test_annotations.values():
    for b in v["boxes"]:
        if b["brand"] != "unknown":
            new_num_test_boxes += 1

new_num_train_boxes2 = 0
for v in new_train_annotations2.values():
    for b in v["boxes"]:
        if b["brand"] != "unknown":
            new_num_train_boxes2 += 1

print(f"Train Boxes: {new_num_train_boxes}, Test Boxes: {new_num_test_boxes}")
print(f"Train Boxes2: {new_num_train_boxes2}")

new_train_annotations = new_train_annotations2


with open('outputs/cleaned_train_annotations.json', 'wb') as f:
    f.write(orjson.dumps(new_train_annotations, option=orjson.OPT_INDENT_2))
with open('outputs/cleaned_test_annotations.json', 'wb') as f:
    f.write(orjson.dumps(new_test_annotations, option=orjson.OPT_INDENT_2))

# Clean up empty folders
for class_name in classes:
    class_path = os.path.join(test_path, class_name)
    for time_dir in os.scandir(class_path):
        if not time_dir.is_dir():
            continue
        if len(os.listdir(time_dir.path)) == 0:
            print(f"Empty folder: {time_dir.path}")
            os.rmdir(time_dir.path)


# Generate distribution plots
def plot_distribution(count_dict, title, filename, xlabel):
    count_dict = dict(sorted(count_dict.items(), key=lambda x: int(x[0])))
    plt.figure(figsize=(10, 5))
    plt.bar(count_dict.keys(), count_dict.values(), alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.savefig(f'outputs/postprocess/{filename}')
    plt.close()

# Count box distributions
train_box_count, test_box_count = {}, {}
train_time_count, test_time_count = {}, {}
for v in tqdm(train_annotations.values(), desc="Counting train boxes"):
    train_box_count[len(v["boxes"])] = train_box_count.get(len(v["boxes"]), 0) + 1
    train_time_count[v["time"]] = train_time_count.get(v["time"], 0) + 1

for v in tqdm(test_annotations.values(), desc="Counting test boxes"):
    test_box_count[len(v["boxes"])] = test_box_count.get(len(v["boxes"]), 0) + 1
    test_time_count[v["time"]] = test_time_count.get(v["time"], 0) + 1

# convert box_count to percentage
train_box_count = {k: v/len(train_annotations) for k, v in train_box_count.items()}
test_box_count = {k: v/len(test_annotations) for k, v in test_box_count.items()}
train_time_count = {k: v/len(train_annotations) for k, v in train_time_count.items()}
test_time_count = {k: v/len(test_annotations) for k, v in test_time_count.items()}

plot_distribution(train_box_count, "Train Box Distribution", "train_box_distribution.png", "Number of Boxes")
plot_distribution(test_box_count, "Test Box Distribution", "test_box_distribution.png", "Number of Boxes")
plot_distribution(train_time_count, "Train Time Distribution", "train_time_distribution.png", "Time")
plot_distribution(test_time_count, "Test Time Distribution", "test_time_distribution.png", "Time")
