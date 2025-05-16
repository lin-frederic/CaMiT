import os
import json
import argparse
from tqdm import tqdm

import cv2
import torch
#from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from torch.utils.data import Dataset

class OutlierDataset(Dataset):
    def __init__(self,  files):
        self.data = []
        for file in files:
            self.data.append(file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
                    {
                        "type": "text", 
                        "text": """
                            Please analyze the image and answer the following two questions:
                            1. Does the image only show the interior of the car with little to no exterior visible?
                                - Answer with true if it is an interior view, otherwise false
                            2. Is the image too zoomed-in, showing only a small portion of the car, making it difficult to recognize its model?
                                - If the image only contains a tiny part of the car (e.g., just a wheel, headlight, or badge) and makes model recognition difficult, answer true.
                                - If the car is not fully visible but still recognizable, answer false.
                            Return your answers in the following JSON format: {"interior": <true/false>, "zoomed_in": <true/false>}
                        """
                    }
                ]
            }
        ] for image_path in image_paths
    ]
    return messages


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, help="Path to the test directory", default="clean_outlier/error_analysis")
    parser.add_argument("--cache_path", type=str, help="Path to cache the annotation results", default="clean_outlier/cache_qwen")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=2)
    parser.add_argument("--workers", type=int, help="Number of workers", default=4)
    args = parser.parse_args()

    test_files = os.listdir(args.test_dir)
    test_files = [os.path.join(args.test_dir, file) for file in test_files]

    dataset = OutlierDataset(test_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
   
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",use_fast=True)
    processor.tokenizer.padding_side = "left"
    outlier_qwen = {}
    for image_paths in dataloader:
        messages = create_messages(image_paths)
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
        for i, (output_text, image_path) in enumerate(zip(output_texts, image_paths)):
            outlier_qwen[os.path.basename(image_path)] = output_text
    
    with open("clean_outlier/outlier_qwen.json", "w") as f:
        json.dump(outlier_qwen, f)
