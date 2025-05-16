import os
import sys
moco_path = os.path.join(os.path.dirname(__file__), 'moco-v3')
if moco_path not in sys.path:
    sys.path.append(moco_path)
    sys.path.append(os.path.join(moco_path, 'moco'))

import moco.builder
import moco.loader

import vits

from functools import partial
import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm

from peft import LoraConfig, get_peft_model, PeftModel


def get_model_without_adapter(model_size, model_path):
    model_name_map = {
            's': 'vit_small',
            'b': 'vit_base'
        }
    model = moco.builder.MoCo_ViT(
        partial(vits.__dict__[model_name_map[model_size]], stop_grad_conv1=True),
        256,4096, 0.2)

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.base_encoder
    model.head = nn.Identity()
    return model

def get_model_with_lora(model_size, model_path):
    model = get_model_without_adapter(model_size, model_path)
    return get_model_with_lora_from_model(model)
 
def get_model_with_lora_from_model(model):
    if isinstance(model, PeftModel):
        print("[Info] Existing LoRA detected: merging and unloading...")
        model = model.merge_and_unload()  # returns a standard nn.Module
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv", "proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model



def get_model(args):
    state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = {k.replace("base_model.base_model.", "base_model."): v for k, v in state_dict.items() if "base_model.base_model" in k}
    model = get_model_with_lora(args.model_size, args.pretrained_model_path)
    """msg = model.load_state_dict(state_dict, strict=False)
    print(msg)"""
    model.load_state_dict(state_dict)
    model_name = f"mocov3_{args.model_size}_lora"
    print(f"Using model: {model_name}")
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224), # Resize shortest side to 224
            torchvision.transforms.CenterCrop(224), # Center crop to 224x224
            torchvision.transforms.ToTensor(), # Convert to tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet stats
        ])
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, transform , model_name, device


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
