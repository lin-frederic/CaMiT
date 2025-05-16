import json
import base64
import os
from tqdm import tqdm
import random
from openai import OpenAI
import argparse

from interactive import encode_image, categorize_system_prompt

def encode_image(image_path):
    """ Convert image to Base64 encoding. """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def write_task(image_path,class_name):
    base64_image = encode_image(image_path)
    task = {
        "custom_id": image_path.split("/")[-1].split(".")[0],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "temperature": 0,
            "response_format":{
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": class_name
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        }
    }
    return task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write batch requests for ChatGPT model.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    parser.add_argument("--images_dir", type=str, help="Directory containing images", default="test_crops")
    parser.add_argument("--class_mapping", type=str, help="Path to class_mapping.json file", default="test_crops/class_mapping.json")
    parser.add_argument("--output", type=str, help="Output file for batch requests", default="chatgpt/batch_requests.jsonl")

    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    images = [f"{args.images_dir}/{image}" for image in os.listdir(args.images_dir) if image.endswith(".jpg")]
    with open(args.class_mapping, "r") as f:
        class_mapping = json.load(f)

    batch_requests = []
    for image_path in tqdm(images, desc="Preparing JSONL"):
        class_name = class_mapping[image_path]
        task = write_task(image_path,class_name)
        batch_requests.append(task)

    with open(args.output, "w") as f:
        for task in batch_requests:
            f.write(json.dumps(task) + "\n")