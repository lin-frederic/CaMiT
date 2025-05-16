import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_adapter import clip_adapter
from clip_without_adapter import get_model_with_lora


def get_model(args):
    state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items() if "base_model" in k}
    model, preprocess_train, preprocess_val = get_model_with_lora(args.model_size)
    #model, preprocess_train, preprocess_val = clip_adapter(args.model_size)
    model.load_state_dict(state_dict, strict=False)

    model_name = f"adapter_clip_{args.model_size}"
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