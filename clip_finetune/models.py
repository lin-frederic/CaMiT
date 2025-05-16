import torch
from open_clip import create_model_and_transforms
from peft import LoraConfig, get_peft_model, PeftModel

def get_model_without_adapter(model_size="b"):
    model_name_map =  {
            'b': 'ViT-B-16',
            'l': 'ViT-L-14',
        }
    pretrained_map = {
        'b':'laion2b_s34b_b88k',
        'l':'laion2b_s32b_b82k',
    }
    model_name = model_name_map[model_size]
    pretrained = pretrained_map[model_size]

    pretrained_model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
    pretrained_model = pretrained_model.visual
    return pretrained_model, preprocess_train, preprocess_val


def get_model_with_lora(model_size="b"):
    model, preprocess_train, preprocess_val = get_model_without_adapter(model_size)
    model = get_model_with_lora_from_model(model)
    return model, preprocess_train, preprocess_val

def get_model_with_lora_from_model(model):
    if isinstance(model, PeftModel):
        print("[Info] Existing LoRA detected: merging and unloading...")
        model = model.merge_and_unload()  # returns a standard nn.Module
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_model_with_lora


def get_model(args):
    state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = {k.replace("base_model.base_model.", "base_model."): v for k, v in state_dict.items() if "base_model.base_model" in k}
    model, preprocess_train, preprocess_val = get_model_with_lora(args.model_size)
    """msg = model.load_state_dict(state_dict, strict=False)
    print(msg)"""
    model.load_state_dict(state_dict)

    model_name = f"clip_{args.model_size}_lora"
    print(f"Using model: {model_name}")

    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, preprocess_val, model_name, device


def get_embeddings(model,dataloader,device, model_name=None):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, _labels in tqdm(dataloader):
            if model_name!="Qwen-2-5-VL-7B-Instruct":
                images = images.to(device)
            _embeddings = model(images).cpu()
            embeddings.append(_embeddings)
            labels += list(_labels)
    embeddings = torch.cat(embeddings, dim=0) # (N,D)
    return embeddings, labels

def load_embeddings(model, dataloader, device, features_path, model_name=None):
    if os.path.exists(features_path):
        features = torch.load(features_path)
        embeddings = features["embeddings"]
        labels = features["labels"]
    else:
        embeddings, labels = get_embeddings(model, dataloader, device, model_name)
        torch.save({"embeddings": embeddings, "labels": labels}, features_path)
    return embeddings, labels

def get_model_dim(model):
    input_tensor = torch.randn(1, 3, 224, 224).to("cuda")
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape[1]  # Return the dimension of the output tensor
