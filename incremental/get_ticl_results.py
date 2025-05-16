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

def rename_method(method_name):
    if "fecam" in method_name:
        method_name = "FeCAM"
    elif "ranpac" in method_name:
        method_name = "RanPAC"
    elif "randumb" in method_name:
        method_name = "RanDumb"
    elif "ncmv2" in method_name:
        method_name = "NCM-TI"
    elif "ncm" in method_name:
        method_name = "NCM"
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    return method_name
    
if __name__ == "__main__":
    model = "mocov3_b_lora"
    results_dirs = ["ncm_results","ncmv2_results", "fecamv3_results","randumb_results", "ranpac_results" ]

    method_heatmap_data = {}
    method_acc = {}
    for results_dir in results_dirs:
        heatmap_data, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap(results_dir, model)
        method_heatmap_data[results_dir] = heatmap_data
        method_acc[results_dir] = {"avg_acc": avg_acc, "current_acc": current_acc, "future_acc": future_acc, "past_acc": past_acc}

    train_years = [x[2:] for x in train_years]
    test_years = [x[2:] for x in test_years]

    subplot_size = 8

    # Step 2: Plot heatmap with the same scale
    os.makedirs(f"final_results", exist_ok=True)
    fig, axes = plt.subplots(1, len(results_dirs), figsize=(subplot_size * len(results_dirs), subplot_size))

    for i, method in enumerate(results_dirs):
        heatmap_data = method_heatmap_data[method]
        sns.heatmap(
            heatmap_data,
            ax=axes[i],
            xticklabels=False,
            yticklabels=False,
            vmin=35,
            vmax=100,
            cmap="viridis",
            cbar=False,
        )
        axes[i].set_title(f"{rename_method(method)}", fontsize=46)
        if i == 0:
            axes[i].set_ylabel("Train Year", fontsize=48)
            ytick = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 4 == 0]
            ytick_labels = [x for x in train_years if (int(x) - 2007) % 4 == 0]
            axes[i].set_yticks(ytick)
            axes[i].set_yticklabels(ytick_labels, fontsize=46)
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])
            axes[i].tick_params(axis='y', which='both', left=False, right=False)

        axes[i].set_xlabel("Test Year", fontsize=48)
        xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 4 == 0]
        xtick_labels = [x for x in test_years if (int(x) - 2007) % 4 == 0]
        axes[i].set_xticks(xtick)
        axes[i].set_xticklabels(xtick_labels, fontsize=46)
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False)
    # Adjust layout to make room for the colorbar
    norm = plt.Normalize(vmin=35, vmax=100)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Add colorbar to the right of the last subplot
    cbar_ax = fig.add_axes([1., 0.16, 0.02, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=46)
    cbar.set_label("Accuracy (%)", fontsize=46, labelpad=20)
    plt.tight_layout()
    plt.savefig(os.path.join("final_results", "ticl_results_heatmap.png"), bbox_inches='tight')
    plt.show()
