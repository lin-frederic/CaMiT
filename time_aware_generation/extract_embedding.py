import sys
import os 
moco_path = os.path.join(os.path.dirname(__file__), 'moco-v3')
if moco_path not in sys.path:
    sys.path.append(moco_path)
    sys.path.append(os.path.join(moco_path, 'moco'))


import sys
import os
import torch
import pickle
import argparse
import vits
import moco.builder
import tqdm
import torchvision.transforms as transforms
from PIL import Image
from functools import partial
import clip
from typing import List
import torch.multiprocessing as mp


MODEL_CHECKPOINTS = {
    "clip_large":   "/home/data/bambaw/CLIP/weights/ViT-L-14/clip-vit-large-patch14.pt",
    "clip_base":    "/home/data/bambaw/CLIP/weights/clip_base/ViT-B-32.pt"
}
SCENARIOS = ["without_year", "with_year"]
METHODS = ["finetuned1", "finetuned2", "plain_sd"]

# Transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def pil_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_moco_model(checkpoint_dir, model_size="vit_small", device="cpu"):
    model = moco.builder.MoCo_ViT(
        partial(vits.__dict__[model_size], stop_grad_conv1=True),
        dim=256, mlp_dim=4096, T=0.2
    )
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_0196.pth.tar")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.base_encoder
    model.head = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def load_clip_model(checkpoint_path, device):
    clip_model, clip_preprocess = clip.load(checkpoint_path, device=device)
    clip_model.eval()
    return clip_model, clip_preprocess


def extract_moco_feats(paths: List[str], model, device: torch.device) -> torch.Tensor:
    feats = []
    for p in tqdm.tqdm(paths, desc="Extracting features", unit="img"):
        img = pil_loader(p)
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
        f = out.mean(dim=1) if out.ndim == 3 else out
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


def extract_clip_feats(paths: List[str], clip_model, clip_preprocess, device: torch.device) -> torch.Tensor:
    feats = []
    for p in tqdm.tqdm(paths, desc="Extracting features..", unit="img"):
        img = clip_preprocess(pil_loader(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


def save_embeddings(embs: torch.Tensor, paths: List[str], file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump((embs, paths), f)


def process_images(model_name: str, method: str, scenario: str,
                   device: torch.device, load_real: bool,
                   local_rank: int, world_size: int):
    model = None
    clip_model = clip_preprocess = None
    ckpt_dir = MODEL_CHECKPOINTS[model_name]
    if model_name == "mocov3_base" or model_name == "joint_pretrain_base":
        model = load_moco_model(ckpt_dir, model_size="vit_base", device=device)
    elif model_name == "mocov3_small" or model_name == "joint_pretrain_small":
        model = load_moco_model(ckpt_dir, model_size="vit_small", device=device)
    else:
        clip_model, clip_preprocess = load_clip_model(ckpt_dir, device=device)

    gen_base = os.path.join(GEN_ROOT, method, scenario)
    real_base = os.path.join(REAL_ROOT, scenario)
    if not os.path.isdir(gen_base):
        print(f"[Rank {local_rank}] Missing generated dir: {gen_base}")
        return

    classes = sorted(os.listdir(gen_base))
    for idx, cls in enumerate(classes):
        if idx % world_size != local_rank:
            continue
        cls_gen = os.path.join(gen_base, cls)
        cls_real = os.path.join(real_base, cls)

        if scenario == "with_year":
            for year in sorted(os.listdir(cls_gen)):
                gen_dir = os.path.join(cls_gen, year)
                real_dir = os.path.join(cls_real, year)
                if not (os.path.isdir(gen_dir) and os.path.isdir(real_dir)):
                    print(f"[Rank {local_rank}] Skipping {cls}/{year}: missing dir")
                    continue
                gen_paths = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir)]
                real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
                print(f"[Rank {local_rank}] {cls}/{year}: {len(real_paths)} real, {len(gen_paths)} gen")
                fx = fy = None
                if model is not None:
                    fx = extract_moco_feats(real_paths, model, device) if not load_real else None
                    fy = extract_moco_feats(gen_paths, model, device)
                else:
                    fx = extract_clip_feats(real_paths, clip_model, clip_preprocess, device) if not load_real else None
                    fy = extract_clip_feats(gen_paths, clip_model, clip_preprocess, device)
                out_dir = os.path.join(EMB_ROOT, model_name, method, scenario, cls, year)
                os.makedirs(out_dir, exist_ok=True)
                if fx is not None:
                    save_embeddings(fx, real_paths, os.path.join(out_dir, 'real_embeddings.pkl'))
                save_embeddings(fy, gen_paths, os.path.join(out_dir, 'gen_embeddings.pkl'))
        else:
            gen_paths = [os.path.join(cls_gen, f) for f in os.listdir(cls_gen)]
            real_paths = [os.path.join(cls_real, f) for f in os.listdir(cls_real)]
            print(f"[Rank {local_rank}] {cls}: {len(real_paths)} real, {len(gen_paths)} gen")
            if model is not None:
                fx = extract_moco_feats(real_paths, model, device) if not load_real else None
                fy = extract_moco_feats(gen_paths, model, device)
            else:
                fx = extract_clip_feats(real_paths, clip_model, clip_preprocess, device) if not load_real else None
                fy = extract_clip_feats(gen_paths, clip_model, clip_preprocess, device)
            out_dir = os.path.join(EMB_ROOT, model_name, method, scenario, cls)
            os.makedirs(out_dir, exist_ok=True)
            if fx is not None:
                save_embeddings(fx, real_paths, os.path.join(out_dir, 'real_embeddings.pkl'))
            save_embeddings(fy, gen_paths, os.path.join(out_dir, 'gen_embeddings.pkl'))
    print(f"[Rank {local_rank}] ✓ Done embeddings for {model_name}/{method}/{scenario}")


def main_worker(local_rank: int, args):
    available = torch.cuda.device_count()
    world_size = args.world_size
    if available < world_size:
        print(f"Only {available} GPUs available, adjusting world_size to {available}")
        world_size = available
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    process_images(
        model_name=args.model_name,
        method=args.method,
        scenario=args.scenario,
        device=device,
        load_real=args.load_real_embeddings,
        local_rank=local_rank,
        world_size=world_size
    )


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings in parallel across GPUs")
    parser.add_argument("model_name", choices=list(MODEL_CHECKPOINTS.keys()),
                        help="Which model to use for extraction")
    parser.add_argument("method", choices=METHODS,
                        help="Generation method (finetuned1, finetuned2, plain_sd)")
    parser.add_argument("scenario", choices=SCENARIOS,
                        help="Scenario (with_year or without_year)")
    parser.add_argument("--load_real_embeddings", action="store_true",
                        help="Skip real image processing if embeddings already exist")
    parser.add_argument("--world_size", type=int, default=8,
                        help="Number of GPUs/processes to use for parallel extraction")
    
    parser.add_argument("--gen_root", type=str, required=True,
                        help="Root directory for generated images")
    parser.add_argument("--real_root", type=str, required=True,
                        help="Root directory for real images")
    parser.add_argument("--emb_root", type=str, required=True,
                        help="Root directory to save embeddings")
    args = parser.parse_args()
    
    global GEN_ROOT, REAL_ROOT, EMB_ROOT
    GEN_ROOT = args.gen_root
    REAL_ROOT = args.real_root
    EMB_ROOT = args.emb_root

    world_size = args.world_size
    mp.spawn(main_worker, args=(args,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()