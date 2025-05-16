import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict
from tqdm import tqdm

def get_pretrain_ViT(pretrained=False, dinov2=False):
    # https://pytorch.org/vision/main/models/vision_transformer.html
    if(dinov2):
        print("[info] Use dinoV2 pretrained model")
        assert pretrained == True, "Error: asking for dinov2 without pretraining."
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    else:
        if(pretrained):
            # https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
            return torchvision.models.vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            return torchvision.models.vit_b_16()
        
def change_classifier_ViT(pytorch_ViT, num_classes):
    """
    for more details see : https://github.com/pytorch/vision/blob/a8a9f2d74918ed446ee7e32e32170765dedd7955/torchvision/models/vision_transformer.py#L237
    """
    if(pytorch_ViT.__class__.__name__ == 'DinoVisionTransformer'):
        pytorch_ViT.head = nn.Linear(pytorch_ViT.hidden_dim, num_classes)
        return pytorch_ViT
    heads_layers = OrderedDict()
    heads_layers["head"] = nn.Linear(pytorch_ViT.hidden_dim, num_classes)
    pytorch_ViT.heads = nn.Sequential(heads_layers)
    return pytorch_ViT

def get_pretrain_ViT_with_new_classifier(num_classes, pretrained=False, dinov2=False):
    """
    add a new head without pre_logits
    for more details see : https://github.com/pytorch/vision/blob/a8a9f2d74918ed446ee7e32e32170765dedd7955/torchvision/models/vision_transformer.py#L237
    """
    model = get_pretrain_ViT(pretrained = pretrained, dinov2=dinov2)
    model = change_classifier_ViT(model, num_classes)
    return model

def remove_classifier_ViT(pytorch_ViT):
    if(pytorch_ViT.__class__.__name__ == 'DinoVisionTransformer'):
        pytorch_ViT.head = nn.Identity()
        return pytorch_ViT
    heads_layers = OrderedDict()
    heads_layers["head"] = nn.Identity()
    pytorch_ViT.heads = nn.Sequential(heads_layers)
    return pytorch_ViT

def get_ViT_without_classifier(pretrained=True, dinov2=False):
    model = get_pretrain_ViT(pretrained=pretrained, dinov2=dinov2)
    model = remove_classifier_ViT(model)
    return model

def get_model(dinov2=True):
    model = get_ViT_without_classifier(pretrained=True, dinov2=dinov2)
    return model

def get_embeddings(model, dataloader, device):
    embeddings = []
    image_paths = []

    with torch.no_grad():
        for images, annotations, paths in tqdm(dataloader): # box annotations should not matter for deduplication
            images = images.to(device)
            _embeddings = model(images).cpu()
            embeddings.append(_embeddings)
            image_paths += list(paths)
    embeddings = torch.cat(embeddings, dim=0) # (N,D)
    return embeddings, image_paths


if __name__ == "__main__":
    model = get_model()
    print(model)