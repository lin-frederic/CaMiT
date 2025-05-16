# Standard library imports
import os
import json
import argparse
import logging

# Third-party libraries
from tqdm import tqdm  # progress bar for loops
import cv2             # image loading and manipulation
import torch
from torch.utils.data import Dataset

# Hugging Face transformers & VLLM imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Custom utility function for vision processing
from qwen_vl_utils import process_vision_info

# Logging setup
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Function to split dataset into subsets
def split_dataset(dataset, num_splits):
    total_samples = len(dataset)
    split_size = total_samples // num_splits
    splits = []
    indices = list(range(total_samples))
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else total_samples
        subset_indices = indices[start:end]
        subset = torch.utils.data.Subset(dataset, subset_indices)
        splits.append(subset)
    return splits

# Custom dataset class to load image paths and boxes
class OutlierDataset(Dataset):
    def __init__(self, annotations):
        self.data = []
        for image_id, image_data in annotations.items():
            image_path = image_data["image_path"]
            boxes = image_data["boxes"]
            for box_id, box in enumerate(boxes):
                if "gpt_class" in box:
                    self.data.append((image_path, box["box"], box_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, box, box_id = self.data[idx]
        image = cv2.imread(image_path)
        return image, image_path, box, box_id

# Function to combine individual samples into a batch
def collate_fn(batch):
    images = [b[0] for b in batch]
    image_paths = [b[1] for b in batch]
    boxes = [b[2] for b in batch]
    box_ids = [b[3] for b in batch]
    return images, image_paths, boxes, box_ids

# Function to generate prompt messages for the vision-language model
def create_messages(image_paths):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": """
                        Please analyze the image and answer the following two questions:
                        1. Does the image only show the interior of the car with little to no exterior visible?
                           - Answer with true if it is an interior view, otherwise false
                        2. Is the image too zoomed-in, showing only a small portion of the car, making it difficult to recognize its model?
                           - If the image only contains a tiny part of the car (e.g., just a wheel, headlight, or badge) and makes model recognition difficult, answer true.
                           - If the car is not fully visible but still recognizable, answer false.
                        Return your answers in the following JSON format: {"interior": <true/false>, "zoomed_in": <true/false>}
                    """}
                ]
            }
        ] for image_path in image_paths
    ]
    return messages

# Save cropped and resized image regions to a local cache
def save_cache(images, image_paths, boxes, box_ids, cache_path):
    output_image_paths = []
    for image, image_path, box, box_id in zip(images, image_paths, boxes, box_ids):
        cx, cy, w, h = box
        x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, (224, 224))
        image_name = os.path.basename(image_path).replace(".jpg", f"_{box_id}.jpg")
        image_path = os.path.join(cache_path, image_name)
        cv2.imwrite(image_path, image)
        output_image_paths.append(image_path)
    return output_image_paths

# Clears previously cached files
def clean_cache(cache_path):
    for file in os.listdir(cache_path):
        os.remove(os.path.join(cache_path, file))

# Main entry point
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, default="outputs/train_gpt_annotations_with_unknown.json")
    parser.add_argument("--test_annotations", type=str, default="outputs/test_gpt_annotations_with_unknown.json")
    parser.add_argument("--cache_path", type=str, default="clean_outlier/cache_qwen")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=16)
    parser.add_argument("--split_id", type=int, default=0)
    args = parser.parse_args()

    # Load JSON annotations
    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)

    # Combine train and test annotations
    all_annotations = {**train_annotations, **test_annotations}

    # Create and split dataset
    dataset = OutlierDataset(all_annotations)
    splits = split_dataset(dataset, args.num_splits)
    current_split = splits[args.split_id]

    # Initialize DataLoader
    dataloader = torch.utils.data.DataLoader(
        current_split, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        collate_fn=collate_fn
    )

    # Load existing results if available
    os.makedirs("clean_outlier/outlier_qwen", exist_ok=True)
    outlier_file = f"clean_outlier/outlier_qwen/split_{args.split_id}.json"
    if os.path.exists(outlier_file):
        with open(outlier_file) as f:
            outlier_qwen = json.load(f)
    else:
        outlier_qwen = {}

    # Load the vision-language model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    processor.tokenizer.padding_side = "left"

    # Prepare image cache folder
    os.makedirs(args.cache_path, exist_ok=True)
    split_cache_path = os.path.join(args.cache_path, f"split_{args.split_id}")
    os.makedirs(split_cache_path, exist_ok=True)
    cache_path = split_cache_path

    # Loop over data batches
    for images, image_paths, boxes, box_ids in tqdm(dataloader):
        # Skip already processed samples
        unprocessed = []
        for i, (image_path, box_id) in enumerate(zip(image_paths, box_ids)):
            box_name = os.path.basename(image_path).replace(".jpg", f"_{box_id}")
            if box_name in outlier_qwen:
                continue
            unprocessed.append(i)
        if len(unprocessed) == 0:
            logger.debug("All images are already annotated")
            continue

        # Filter unprocessed samples
        images = [images[i] for i in unprocessed]
        image_paths = [image_paths[i] for i in unprocessed]
        boxes = [boxes[i] for i in unprocessed]
        box_ids = [box_ids[i] for i in unprocessed]

        # Save cropped images to cache
        output_image_paths = save_cache(images, image_paths, boxes, box_ids, cache_path)
        
        # Generate prompts/messages
        messages = create_messages(output_image_paths)
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # Prepare inputs for model
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        # Extract and decode outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Store results
        for i, (output_text, image_path, box_id) in enumerate(zip(output_texts, image_paths, box_ids)):
            output_text = output_text.strip()
            box_name = os.path.basename(image_path).replace(".jpg", f"_{box_id}")
            outlier_qwen[box_name] = output_text

        # Clean cache folder and save output
        clean_cache(cache_path)
        with open(outlier_file, "w") as f:
            json.dump(outlier_qwen, f)
