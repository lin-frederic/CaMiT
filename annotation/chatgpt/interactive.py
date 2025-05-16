import os
import json
import base64
from openai import OpenAI
import argparse
import random
from tqdm import tqdm

categorize_system_prompt = '''Analyze the image and return:
{
    "model": string,  // Identified car model (return provided class if it matches)
    "model_probability": number,  // Probability (0-100) of the predicted model
    "car_probability": number  // Probability (0-100) that the car is real
}
Strict JSON format required'''

def encode_image(image_path):
    """ Convert image to Base64 encoding. """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def predict(image_path, class_name):
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
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
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive script to test the ChatGPT model.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    parser.add_argument("--images_dir", type=str, help="Directory containing images", default="test_crops")
    parser.add_argument("--class_mapping", type=str, help="Path to class_mapping.json file", default="test_crops/class_mapping.json")
    parser.add_argument("--N", type=int, help="Number of images to process", default=10)
    parser.add_argument("--results", type=str, help="Path to save results", default="interactive_results.json")
    args = parser.parse_args()


    client = OpenAI(api_key=args.api_key)
    images = [f"{args.images_dir}/{image}" for image in os.listdir(args.images_dir) if image.endswith(".jpg")]
    # shuffle images
    random.shuffle(images)
    images = images[:args.N]

    with open(args.class_mapping, "r") as f:
        class_mapping = json.load(f)

    results = {}
    for image_path in tqdm(images):
        print(f"Image: {image_path}")
        class_name = class_mapping.get(image_path, "") # class name is predicted by Qwen
        print(f"Class: {class_name}")
        result = predict(image_path, class_name)
        results[image_path] = {"gpt-4o": result, "qwen": class_name} 
        print(f"Result: {result}")

    with open(args.results, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)