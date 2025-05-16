import torch
import torch.nn as nn
import torchvision
import open_clip

import os
from collections import OrderedDict
from functools import partial
from tqdm import tqdm

import sys
moco_path = os.path.join(os.path.dirname(__file__), 'moco-v3')
if moco_path not in sys.path:
    sys.path.append(moco_path)
    sys.path.append(os.path.join(moco_path, 'moco'))

import moco.builder
import moco.loader

import vits

from qwen_vision_encoder import Qwen2_5_VisionEncoder

def get_pretrain_ViT(model="dinov2_b", car_mocov3_dir='checkpoints'):
    # https://pytorch.org/vision/main/models/vision_transformer.html
    if "dinov2" in model:
        model_size = model.split("_")[1]
        print("[info] Use dinov2_vit{}14".format(model_size))
        return torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
    elif "clip" in model:
        model_size = model.split("_")[1]
        model_name_map =  {
            'b': 'ViT-B-16',
            'l': 'ViT-L-14',
            'h': 'ViT-H-14',
            'g': 'ViT-G-14'
        }
        pretrained_map = {
            'b':'laion2b_s34b_b88k',
            'l':'laion2b_s32b_b82k',
            'h':'laion2b_s32b_b79k',
            'g':'laion2b_s34b_b88k'
        }
        model_name = model_name_map[model_size]
        pretrained = pretrained_map[model_size]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        return [model, preprocess]

    elif "qwen" in model:
        model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct")
        model = Qwen2_5_VisionEncoder(model_path, device='cuda')
        return model

    elif "car_mocov3" in model:
        model_size = model.split("_")[2]
        epoch = int(model.split("_")[3])
        model_name_map = {
            's': 'vit_small',
            'b': 'vit_base'
        }
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[model_name_map[model_size]], stop_grad_conv1=True),
            256,4096, 0.2)
        """model = moco.builder.MoCo_ViT(
            partial(vits.__dict__['vit_small'], stop_grad_conv1=True),
            256,4096, 0.2)
        """
        
        # car_mocov3_epoch : 50, 100, 150, 200, 250, 299 --> 0050, 0100, 0150, 0200, 0250, 0299
        checkpoint = os.path.join(car_mocov3_dir, f"checkpoint_{epoch:04d}.pth.tar")
        checkpoint = torch.load(checkpoint, map_location='cpu')
        # load the model state dict, remove the prefix 'module.'
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model = model.base_encoder
        model.head = nn.Identity()
        return model
    elif "mocov3" in model:
        model_size = model.split("_")[1]
        model_name_map = {
            's': 'vit_small',
            'b': 'vit_base'
        }

        model_checkpoint_map = {
            's': 'vit-s-300ep.pth',
            'b': 'vit-b-300ep.pth'
        }

        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[model_name_map[model_size]], stop_grad_conv1=True),
            256,4096, 0.2)
        

        print("[info] Use ViT{}_mocov3".format(model_size))

        checkpoint_dir = torch.hub.get_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints/mocov3')
        checkpoint = torch.load(os.path.join(checkpoint_dir, model_checkpoint_map[model_size]), map_location='cpu')
        # load the model state dict, remove the prefix 'module.'
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model = model.base_encoder
        model.head = nn.Identity()
        return model
        
    elif "mocov3" in model:
        model_size = model.split("_")[1]
        model_name_map = {
            's': 'vit_small',
            'b': 'vit_base'
        }

        model_checkpoint_map = {
            's': 'vit-s-300ep.pth',
            'b': 'vit-b-300ep.pth'
        }

        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[model_name_map[model_size]], stop_grad_conv1=True),
            256,4096, 0.2)
        

        print("[info] Use ViT{}_mocov3".format(model_size))

        checkpoint_dir = torch.hub.get_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints/mocov3')
        checkpoint = torch.load(os.path.join(checkpoint_dir, model_checkpoint_map[model_size]), map_location='cpu')
        # load the model state dict, remove the prefix 'module.'
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model = model.base_encoder
        model.head = nn.Identity()
        return model
    else:
        raise ValueError("Error: model should be in ['dinov2', 'clip', 'mocov3', 'car_mocov3']")
    
def remove_classifier_ViT(pytorch_ViT):
    if(pytorch_ViT.__class__.__name__ == 'DinoVisionTransformer'):
        pytorch_ViT.head = nn.Identity()
        return pytorch_ViT
    elif isinstance(pytorch_ViT, list): # open_clip
        pytorch_ViT, preprocess = pytorch_ViT
        pytorch_ViT.forward = lambda x: pytorch_ViT.encode_image(x)
        return [pytorch_ViT, preprocess]    
    return pytorch_ViT


def get_ViT_without_classifier(model, car_mocov3_dir):
    model = get_pretrain_ViT(model, car_mocov3_dir)
    model = remove_classifier_ViT(model)
    return model


def get_model(args):
    model = args.model
    car_mocov3_dir = args.car_mocov3_dir

    if "car_mocov3" in model:
        args.model = f"car_mocov3_{args.model_size}_{args.epoch}"
    elif "qwen" in model:
        args.model = "qwen"
    else:
        args.model = f"{args.model}_{args.model_size}"

    model = get_ViT_without_classifier(args.model, args.car_mocov3_dir)

    if "qwen" in args.model:
        model_name = "Qwen-2-5-VL-7B-Instruct"
    else:
        model_name = args.model

    print(f"Using model: {model_name}")
    
    if "clip" in args.model:
        model, preprocess = model
        transform = preprocess
        
    elif "qwen" in args.model:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224)
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224), # Resize shortest side to 224
            torchvision.transforms.CenterCrop(224), # Center crop to 224x224
            torchvision.transforms.ToTensor(), # Convert to tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet stats
        ])
    print(f"Transform: {transform}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, transform, model_name, device

class LinearProbingModel(nn.Module):
    def __init__(self, output_dim=768, num_classes=1000):
        super(LinearProbingModel, self).__init__()
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x
    
    def update_classifier(self, num_classes):
        # Extend the classifier to the new number of classes
        old_classifier = self.classifier
        old_num_classes = old_classifier.out_features
        new_classifier = nn.Linear(old_classifier.in_features, num_classes)
        new_classifier.weight.data[:old_num_classes] = old_classifier.weight.data
        new_classifier.bias.data[:old_num_classes] = old_classifier.bias.data
        self.classifier = new_classifier
    
def get_model_dim(model):
    input_tensor = torch.randn(1, 3, 224, 224).to("cuda")
    with torch.no_grad():
        output = model(input_tensor)
    return output.shape[1]  # Return the dimension of the output tensor

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

if __name__ == "__main__":
    model = get_model(dinov2=True,model_size='b')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input.to(device))
    print(output.shape)  # Should be (1, 768) for ViT-B-16
