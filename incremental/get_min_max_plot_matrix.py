import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def results_to_heatmap(results_dir, model_name):
    results_dir = os.path.join(results_dir, model_name)
    group_results = {}
    # Load JSON files
    for filename in tqdm(os.listdir(results_dir), desc="Processing files"):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r") as f:
                data = json.load(f)
            group_results = {**group_results, **data}  # Merge dictionaries

    # Sort train and test years
    train_years = sorted(group_results.keys(), key=lambda x: int(x))
    test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))
    
    heatmap_data = np.zeros((len(train_years), len(test_years)))
    avg_acc = 0
    past_acc = 0
    future_acc = 0
    current_acc = 0

    for i, train_year in enumerate(train_years):
        for j, test_year in enumerate(test_years):
            heatmap_data[i, j] = round(group_results[train_year][test_year] * 100, 2)
            avg_acc += group_results[train_year][test_year]
            if i < j:
                future_acc += group_results[train_year][test_year]
            elif i > j:
                past_acc += group_results[train_year][test_year]
            else:
                current_acc += group_results[train_year][test_year]

    avg_acc /= (len(train_years) * len(test_years))
    off_diagonal_count = len(train_years) * (len(train_years) - 1)/2
    future_acc /= off_diagonal_count
    past_acc /= off_diagonal_count
    current_acc /= len(train_years)
    avg_acc = round(avg_acc * 100, 2)
    future_acc = round(future_acc * 100, 2)
    past_acc = round(past_acc * 100, 2)
    current_acc = round(current_acc * 100, 2)

    return heatmap_data, train_years, test_years, avg_acc, future_acc, past_acc, current_acc

def rename_model(model_name):
    if "dinov2" in model_name:
        model_size = model_name.split("_")[-1]
        model_name = f"DinoV2-{model_size}"
    elif "clip" in model_name:
        model_size = model_name.split("_")[1]
        if "lora" in model_name:
            model_name = f"CLIP-{model_size} + LoRA"
        else:
            model_name = f"CLIP-{model_size}"
    elif "car_mocov3" in model_name:
        model_size = model_name.split("_")[2]
        model_name = f"MoCoV3-{model_size}"
    elif "mocov3" in model_name:
        model_size = model_name.split("_")[1]
        model_name = f"MoCoV3-{model_size} + LoRA"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_name

if __name__ == "__main__":
    results_dir = "ncm_results"
    plotted_models = ["car_mocov3_s_199", "mocov3_s_lora", "clip_b", "clip_b_lora", "car_mocov3_b_199", "mocov3_b_lora"]
    
    years = [str(x) for x in range(2008, 2024)]
    for year in years:
        plotted_models.append(f"car_mocov3_s_259_{year}")
        plotted_models.append(f"mocov3_s_lora_{year}")
        plotted_models.append(f"mocov3_s_lora_{year}_replay")
    
    # get min and max over all models

    min_acc = 100
    max_acc = 0

    for model_name in tqdm(plotted_models, desc="Processing models"):
        heatmap_data, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap(results_dir, model_name)
        min_acc = min(min_acc, np.min(heatmap_data))
        max_acc = max(max_acc, np.max(heatmap_data))

    model = "mocov3_b_lora"
    results_dirs = ["ncmv2_results", "fecamv3_results", "ranpac_results", "randumb_results"]
    for results_dir in results_dirs:
        heatmap_data, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap(results_dir, model)
        min_acc = min(min_acc, np.min(heatmap_data))
        max_acc = max(max_acc, np.max(heatmap_data))

    print(f"Min accuracy: {min_acc}") # 37.74
    print(f"Max accuracy: {max_acc}") # 98.09