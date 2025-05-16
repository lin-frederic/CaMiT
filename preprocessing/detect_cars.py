import os
import torch
import torch.version
from tqdm import tqdm
from ultralytics import YOLO
import json
from utils import filter_corrupted_images

def filter_car_images(data_path, dest_path, classes, model, vehicle_labels, n_images=0, batch_size=64):
    os.makedirs(dest_path, exist_ok=True)
    with tqdm(total=len(classes)) as outer_bar:
        for class_name in classes:

            # Set up for the class
            class_images = os.listdir(os.path.join(data_path, class_name))
            if n_images > 0:
                class_images = class_images[:n_images]
            class_images = [os.path.join(data_path, class_name, x) for x in class_images]
            #class_images = [os.path.join(data_path, class_name, class_name, x) for x in class_images]
            outer_bar.set_description(f"Processing {len(class_images)} images from class {class_name}")
            os.makedirs(os.path.join(dest_path, class_name), exist_ok=True)
            #os.makedirs(os.path.join(dest_path, class_name, "images"), exist_ok=True)

            # check if annotations already exist to avoid reprocessing
            if os.path.exists(os.path.join(dest_path, class_name, "detections.json")):
                try:
                    with open(os.path.join(dest_path, class_name, "detections.json"), "r") as f:
                        annotations = json.load(f)
                except:
                    annotations = {}
            else:
                annotations = {}
            print(f"Already processed {len(annotations)} images from class {class_name}")
            # Inference
            with tqdm(total=len(class_images)) as inner_bar:
                for b in range(0, len(class_images), batch_size):
                    if b+batch_size > len(class_images): # last batch might be smaller than batch_size
                        batch = class_images[b:]
                    else:
                        batch = class_images[b:b+batch_size]
                    
                    # Check already processed images
                    batch = [x for x in batch if os.path.basename(x) not in annotations]
                    if len(batch) == 0: # avoid reprocessing, before filtering corrupted images (avoid opening images twice)
                        inner_bar.update(batch_size)
                        continue
                    batch = filter_corrupted_images(batch) # effective batch size might be smaller than batch_size if there are corrupted images
                    if len(batch) == 0: # Filtering corrupted images might remove all images
                        inner_bar.update(batch_size)
                        continue
                    with torch.no_grad():
                        results = model.predict(batch, batch=batch_size,stream=True,verbose=False)

                    # Save images and annotations
                    try:
                        for i, result in enumerate(results):
                            image_name = os.path.basename(batch[i])
                            detected_labels = result.boxes.cls

                            # Save image if it contains a car
                            if any([label in vehicle_labels for label in detected_labels]):
                                #image_path = os.path.join(dest_path, class_name, "images", image_name)
                                #result.save(image_path) # don't actually need to save the image, the annotations are enough, remove this later

                                detection_scores = result.boxes.conf 
                                detected_boxes = result.boxes.xywh

                                annotations[image_name] = [{"box": box.tolist(), "label": label.item(), "score": score.item()} for box, label, score in zip(detected_boxes, detected_labels, detection_scores)]
                    except Exception as e:
                        print(f"Error in batch {b} of class {class_name}: {e}")
                        print(f"Failed images: {batch}")
                        exit() # should not happen (solved)
                    inner_bar.update(batch_size)
                    with open(os.path.join(dest_path, class_name, "detections.json"), "w") as f:
                        json.dump(annotations, f)


            outer_bar.update(1) # update progress bar (class)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect cars in images")
    parser.add_argument("--data_path", type=str, help="Path to data", default=os.path.join(os.environ['HOME'], 'cars_images_2019'))
    parser.add_argument("--dest_path", type=str, help="Path to save images", default=os.path.join(os.environ['HOME'], 'cars_detections_2019'))
    parser.add_argument("--class_names", type=str, nargs="+", help="List of class names", default=[])
    parser.add_argument("--n_images", type=int, help="Number of images to process", default=0) # 0 means all images
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--model", type=str, help="Model to use", default="yolo11x.pt")
    args = parser.parse_args()

    data_path = args.data_path
    dest_path = args.dest_path
    
    if not os.path.exists(data_path):
        print(f"Source path {data_path} does not exist. Exiting...")
        exit()

    if len(args.class_names) == 0:
        args.class_names = os.listdir(data_path) # process all classes
        args.class_names.sort()

    for class_name in args.class_names:
        if not os.path.exists(os.path.join(data_path, class_name)):
            print(f"Class {class_name} does not exist in source path. Exiting...")
            exit()
    print("all classes exist, continuing...")
    print(f"Processing classes: {args.class_names}")

    os.makedirs(dest_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit()
    else:
        print(f"Using device: {device}")
    
    model = YOLO(args.model)
    model.to(device)

    coco_labels = model.names # dict id: name
    vehicle_labels = ['car', 'bus', 'train', 'truck']
    vehicle_labels = [x for x in coco_labels if coco_labels[x] in vehicle_labels]

    filter_car_images(data_path, dest_path, args.class_names, model, vehicle_labels, args.n_images, args.batch_size)


    