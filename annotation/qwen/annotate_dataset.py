import os
import argparse
import torch
import json
import re
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from dataset_annotation import CarsDataset, cars_collate_fn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
def create_messages(image_paths):
    messages = [
        [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", 
                 "text": "Please identify the brand and model of the car in the image. If the car is unrecognizable, respond with 'unknown'. The response should be in JSON format as follows: {'brand': '...', 'model': '...'}"
                }
            ]
        }]
        for image_path in image_paths
    ]
    return messages


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection_path", type=str, help="Path to the detection results", default=os.path.join(os.environ['HOME'], 'cars_detections'))
    parser.add_argument("--deduplication_path", type=str, help="Path to the deduplication results", default=os.path.join(os.environ['HOME'], 'cars_deduplicated'))
    parser.add_argument("--annotation_path", type=str, help="Path to save the annotation results", default=os.path.join(os.environ['HOME'], 'cars_annotations'))
    parser.add_argument("--cache_path", type=str, help="Path to cache the annotation results", default=os.path.join(os.environ['HOME'], 'cars_annotations_cache'))
    parser.add_argument("--class_names", type=str, nargs="+", help="List of class names", default=["nissan"])
    parser.add_argument("--batch_size", type=int, help="Batch size", default=80)
    args = parser.parse_args()

    detection_path = args.detection_path
    deduplication_path = args.deduplication_path
    annotation_path = args.annotation_path
    annotation_cache_path = args.cache_path
    class_names = args.class_names

    for class_name in class_names:
        assert os.path.exists(os.path.join(deduplication_path, class_name, "unique_images.txt")), f"unique_images.txt not found in {deduplication_path}/{class_name}"
        assert os.path.exists(os.path.join(detection_path, class_name, "detections.json")), f"detections.json not found in {detection_path}/{class_name}"

    logger.info("All classes exist, ready to annotate the dataset")

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")

    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(annotation_cache_path, exist_ok=True)

    for class_name in class_names:
        logger.info(f"Annotating class {class_name}")
        os.makedirs(os.path.join(annotation_path, class_name), exist_ok=True)
        dataset = CarsDataset(detection_path, deduplication_path, class_name)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=cars_collate_fn)
    
        total = len(dataloader)
        logger.info(f"Total number of batches: {total}")
        if os.path.exists(os.path.join(annotation_path, class_name, "annotations.json")):
            with open(os.path.join(annotation_path, class_name, "annotations.json"), "r") as f:
                annotations = json.load(f)
        else:
            annotations = {}

        with tqdm(total=total) as pbar:
            for (image_paths, boxes, labels, scores) in dataloader:
                
                # Check if the image has already been annotated


                image_names = [os.path.basename(image_path) for image_path in image_paths]

                crop_names = dataset.save_crops_batch(image_paths, boxes, os.path.join(annotation_cache_path, class_name))
                crop_paths = [os.path.join(annotation_cache_path, class_name, crop_name) for crop_name in crop_names]

                    
                messages = create_messages(crop_paths)
                texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for i in range(len(image_paths)):
                    image_name = os.path.basename(image_paths[i])
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]

                    if image_name not in annotations:
                        annotations[image_name] = []
                    
                    annotation_dict = {'box': box, 'label': label, 'score': score}
                    
                    text = output_texts[i]
                    if 'unknown' in text:
                        parsed_dict = {'brand': 'unknown', 'model': 'unknown'}
                    else:
                        clean_text = re.sub(r'```json\n|\n```', '', text)
                        try:
                            parsed_dict = json.loads(clean_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON: {clean_text}, {e}") 
                            parsed_dict = {'brand': 'unknown', 'model': 'unknown'} 
                    for key in parsed_dict:
                        annotation_dict[key] = parsed_dict[key] 
                    annotation_dict['text'] = text
                    annotations[image_name].append(annotation_dict)


                # Delete crops
                for crop_path in crop_paths:
                    os.remove(crop_path)

                # Save annotations
                with open(os.path.join(annotation_path, class_name, "annotations.json"), "w") as f:
                    json.dump(annotations, f, indent=4)
                    
                pbar.update(1)









                


    





