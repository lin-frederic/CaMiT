import os
import glob
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from safetensors.torch import load_file
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import torch.nn as nn

class Qwen2_5_VisionEncoder(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        model_path = os.path.join(model_path, "snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4")
        all_ckpt_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        visual_weights = {}
        for ckpt in all_ckpt_files:
            print(f"Loading {ckpt}...")
            weights = load_file(ckpt)
            for key, val in weights.items():
                if key.startswith("visual."):
                    visual_weights[key.replace("visual.", "")] = val
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.vision_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config=config.vision_config
        )
        self.vision_model.load_state_dict(visual_weights, strict=False)
        self.vision_model.to(device)
        self.vision_model.eval()

        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        # self.processor = AutoProcessor("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        # self.processor.save_pretrained(model_path)

    def forward(self, images):
        texts = [""] * len(images)
        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)
        pixel_values = inputs["pixel_values"].type(torch.bfloat16)
        grid_thw = inputs["image_grid_thw"]
        
        with torch.no_grad():
            batch_embeds = self.vision_model(pixel_values, grid_thw=inputs["image_grid_thw"])
        tokens_per_image = grid_thw[:,0] * grid_thw[:,1] * grid_thw[:,2] # assume all images are the same size
        merge_length = self.processor.image_processor.merge_size**2
        tokens_per_image = tokens_per_image // merge_length
        split_embeddings = torch.split(batch_embeds, tokens_per_image.tolist(), dim=0) # tuple of tensors
        # convert to tensors assuming all images are the same size
        embeddings = list(split_embeddings)
        embeddings = torch.stack(split_embeddings, dim=0) # (batch_size, num_patches, hidden_size)
        
        # average the embeddings over the patches
        embeddings = embeddings.mean(dim=1) # (batch_size, hidden_size)
        return embeddings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load Qwen Vision Encoder")
    parser.add_argument("--model_path", type=str, help="Path to the model directory", default=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct")) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = Qwen2_5_VisionEncoder(args.model_path, device)
    print("Model loaded successfully.")
    print("Vision model:", vision_encoder.vision_model)
    print("Processor:", vision_encoder.processor)

    #image_path = "cars_dataset/test_blurred/2010/5269825264.jpg"
    image_path = "cars_dataset/test_blurred/2023/52757831875.jpg"
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    batch_size = 8
    # repeat the image to create a batch
    images = [image] * batch_size # (8, 3, 224, 224)

    image_embeds = vision_encoder(images)
    print("Image embeddings shape:", image_embeds.shape)