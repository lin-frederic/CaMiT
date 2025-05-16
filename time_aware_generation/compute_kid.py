import os
import pickle
import argparse
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import csv
import torch
import torch.serialization
from torch.serialization import default_restore_location as _orig_default_restore_location

def _map_to_cpu(storage, location):
    return _orig_default_restore_location(storage, 'cpu')

torch.serialization.default_restore_location = _map_to_cpu
from sklearn.metrics.pairwise import rbf_kernel

SCENARIOS = ["without_year","with_year"]
METHODS = ["finetuned1", "finetuned2", "plain_sd"]

MODEL_NAMES = ["clip_large", "clip_base"]
CSV_OUTPUT_PATH = "sdxl_kid_results_clip_base_main.csv"


def load_embeddings(model_name: str, method: str, scenario: str, cls: str, year: str = None):
    emb_dir = os.path.join(EMB_ROOT, model_name, method, scenario, cls)
    if year:
        emb_dir = os.path.join(emb_dir, year)
    print(f"Loading embeddings from: {emb_dir}")

    real_pkl = os.path.join(emb_dir, 'real_embeddings.pkl')
    gen_pkl  = os.path.join(emb_dir, 'gen_embeddings.pkl')
    if os.path.exists(real_pkl) and os.path.exists(gen_pkl):
        with open(real_pkl, 'rb') as f:
            real_embeddings, _ = pickle.load(f)
        with open(gen_pkl, 'rb') as f:
            gen_embeddings, _ = pickle.load(f)
    else:

        real_pt = os.path.join(emb_dir, 'real_embeddings.pt')
        gen_pt  = os.path.join(emb_dir, 'gen_embeddings.pt')
        if not os.path.exists(real_pt) or not os.path.exists(gen_pt):
            raise FileNotFoundError(f"Missing embeddings for {model_name}/{method}/{scenario}/{cls}/{year or ''}")
        real_embeddings = torch.load(real_pt, map_location=torch.device('cpu'), weights_only=True)
        gen_embeddings  = torch.load(gen_pt,  map_location=torch.device('cpu'), weights_only=True)


    if not isinstance(real_embeddings, torch.Tensor):
        real_embeddings = torch.tensor(real_embeddings)
    if not isinstance(gen_embeddings, torch.Tensor):
        gen_embeddings = torch.tensor(gen_embeddings)

    return real_embeddings, gen_embeddings


def compute_kid(real_embeddings: torch.Tensor, gen_embeddings: torch.Tensor) -> float:
    real_norm = real_embeddings / real_embeddings.norm(dim=-1, keepdim=True)
    gen_norm  = gen_embeddings  / gen_embeddings.norm(dim=-1, keepdim=True)

    print("Sample real:", real_norm[0][:5])
    print("Sample gen:", gen_norm[0][:5])

    dim = real_norm.size(1)
    gamma = 1.0 / dim

    R = real_norm.cpu().numpy()
    G = gen_norm.cpu().numpy()
    K_rr = rbf_kernel(R, R, gamma=gamma)
    K_gg = rbf_kernel(G, G, gamma=gamma)
    K_rg = rbf_kernel(R, G, gamma=gamma)

    mmd2 = np.mean(K_rr) + np.mean(K_gg) - 2 * np.mean(K_rg)
    return float(mmd2)


def compute_all_kids(model_name: str) -> dict:
    results = {}

    for method in METHODS:
        for scenario in SCENARIOS:
            scenario_dir = os.path.join(EMB_ROOT, model_name, method, scenario)
            if not os.path.isdir(scenario_dir):
                continue
            for cls in sorted(os.listdir(scenario_dir)):
                cls_path = os.path.join(scenario_dir, cls)
                if scenario == 'with_year':
                    for year in sorted(os.listdir(cls_path)):
                        try:
                            real, gen = load_embeddings(model_name, method, scenario, cls, year)
                            key_str = f"{method}/{scenario}/{cls}"
                            print(f"Computing KID for {key_str} - Real: {real.shape}, Gen: {gen.shape}")
                            print("Real sum:", real.sum().item(), "Gen sum:", gen.sum().item())
                        except FileNotFoundError:
                            continue
                        gen = torch.randn_like(gen)
                        kid_val = compute_kid(real, gen)
                        results[(model_name, method, scenario, cls, year)] = kid_val
                else:
                    try:
                        real, gen = load_embeddings(model_name, method, scenario, cls)
                        key_str = f"{method}/{scenario}/{cls}"
                        print(f"Computing KID for {key_str} - Real: {real.shape}, Gen: {gen.shape}")
                        print("Real sum:", real.sum().item(), "Gen sum:", gen.sum().item())
                    except FileNotFoundError:
                        continue
                    gen = torch.randn_like(gen)
                    kid_val = compute_kid(real, gen)
                    results[(model_name, method, scenario, cls)] = kid_val

    return results

def save_mean_kid_to_csv(results: dict, output_path: str = "mean_kid_results_main.csv"):
    mean_kid = {}
    for key, kid in results.items():
        _, method, scenario, *_ = key
        k = (method, scenario)
        mean_kid.setdefault(k, []).append(kid)

 
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['method', 'scenario', 'mean_KID'])
        writer.writeheader()
        for (method, scenario), values in mean_kid.items():
            writer.writerow({
                'method': method,
                'scenario': scenario,
                'mean_KID': np.mean(values)
            })

    print(f"Saved mean KID scores to {output_path}")

def save_results_to_csv(results: dict, output_path: str = CSV_OUTPUT_PATH):
    fieldnames = ['model_name', 'method', 'scenario', 'class', 'year', 'KID_value']
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, kid in results.items():
            model, method, scenario, cls, *year = key
            writer.writerow({
                'model_name': model,
                'method': method,
                'scenario': scenario,
                'class': cls,
                'year': year[0] if year else '',
                'KID_value': kid
            })
    print(f"Saved per-class KID results to {output_path}")


def plot_mean_kid(results: dict, model_name: str):
    mean_kid = {m: {s: [] for s in SCENARIOS} for m in METHODS}
    for key, kid in results.items():
        _, method, scenario, *_ = key
        if method in mean_kid:
            mean_kid[method][scenario].append(kid)

 
    methods = METHODS
    with_vals = [np.mean(mean_kid[m]['with_year']) if mean_kid[m]['with_year'] else 0 for m in methods]
    without_vals = [np.mean(mean_kid[m]['without_year']) if mean_kid[m]['without_year'] else 0 for m in methods]


    x = np.arange(len(methods))
    width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, with_vals, width, label='With Year')
    plt.bar(x + width/2, without_vals, width, label='Without Year')
    plt.xticks(x, methods)
    plt.ylabel('Mean KID')
    plt.title(f'SDXL Mean KID by Method & Scenario ({model_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sdxl_kid_comparison_main_{model_name}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute KID for models including joint pretrain variants")
    parser.add_argument('model_name', choices=MODEL_NAMES, help='Model to evaluate')
    parser.add_argument('--emb_root', type=str, required=True,help='Root directory path where embeddings are stored')
    args = parser.parse_args()

    global EMB_ROOT
    EMB_ROOT = args.emb_root

    model = args.model_name
    print(f"Computing KID for {model}...")
    results = compute_all_kids(model)
    save_results_to_csv(results)
    save_mean_kid_to_csv(results)  
    plot_mean_kid(results, model)


if __name__ == '__main__':
    main()
