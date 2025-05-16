import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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
            model_name = f"CLIP + L$_i$"
        else:
            model_name = f"CLIP"
    elif "car_mocov3" in model_name:
        model_size = model_name.split("_")[2]
        model_name = f"MoCo v3"
    elif "mocov3" in model_name:
        model_size = model_name.split("_")[1]
        model_name = f"MoCo v3 + L$_i$"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_name


if __name__ == "__main__":
    results_dir = "ncm_results"
    plot_models = ["car_mocov3_s_199", "mocov3_s_lora", "clip_b", "clip_b_lora", "car_mocov3_b_199", "mocov3_b_lora"]
    # Step 1: Collect all accuracy data to determine global min and max
    model_heatmap_data = {}

    model_acc = {}

    for model in plot_models:
        heatmap_data, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap(results_dir, model)
        model_heatmap_data[model] = heatmap_data
        model_acc[model] = {"avg_acc": avg_acc, "current_acc": current_acc, "future_acc": future_acc, "past_acc": past_acc}

    train_years = [x[2:] for x in train_years]
    test_years = [x[2:] for x in test_years]

    subplot_size = 6

    # Step 2: Plot heatmap with the same scale
    os.makedirs(f"final_results", exist_ok=True)
    fig, axes = plt.subplots(1, len(plot_models), figsize=(subplot_size * len(plot_models), subplot_size))

    for i, model in enumerate(plot_models):
        heatmap_data = model_heatmap_data[model]
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
        axes[i].set_aspect("equal")
        """model_title = rename_model(model)
        title_lines = model_title.count("\n") + 1
        y_pos = 1.5 if title_lines > 1 else 1.3
        axes[i].text(
            0.5, y_pos,
            model_title,
            fontsize=46,
            ha="center",
            va="top",
            transform=axes[i].transAxes
        )"""
        axes[i].set_title(f"{rename_model(model)}", fontsize=46, pad=20)
        if i == 0:
            axes[i].set_ylabel("Train Year", fontsize=46)
            ytick = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 8 == 0]
            ytick_labels = [x for x in train_years if (int(x) - 2007) % 8 == 0]
            axes[i].set_yticks(ytick)
            axes[i].set_yticklabels(ytick_labels, fontsize=46)
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])
            axes[i].tick_params(axis='y', which='both', left=False, right=False)

        axes[i].set_xlabel("Test Year", fontsize=46)
        xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 8 == 0]
        xtick_labels = [x for x in test_years if (int(x) - 2007) % 8 == 0]
        axes[i].set_xticks(xtick)
        axes[i].set_xticklabels(xtick_labels, fontsize=46)
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False)

    # === Add vertical line between ViT-S and ViT-B ===
    line = Line2D([0.379,0.379], [-0.15, 0.95], transform=fig.transFigure, color='black', linewidth=4)
    fig.add_artist(line)

    # === Add text labels above ViT-S and ViT-B groups ===
    fig.text(0.245, -0.2, "ViT-S", fontsize=52, ha='center', va='bottom')
    fig.text(0.645, -0.2, "ViT-B", fontsize=52, ha='center', va='bottom')


    #fig.text(0.5, -0.07, "Test Year", ha="center", fontsize=46)
    #plt.subplots_adjust(wspace=0.00, hspace=0.1, right=0.8, left=0.05,bottom=0.05, top=0.95)  # Shrink plot area to leave space for colorbar
    norm = plt.Normalize(vmin=35, vmax=100)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Add colorbar to the right of the last subplot
    cbar_ax = fig.add_axes([0.92, 0.17, 0.02, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=46)
    cbar.set_label("Accuracy", fontsize=46, labelpad=20)
    #plt.tight_layout()
        # Adjust layout to make room for the colorbar
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(os.path.join("final_results", "static_results_heatmap.png"), bbox_inches='tight')
    plt.show()


    # Step 3: Create table with accuracies
    table_data = []
    table_headers = ["Model", "Avg Acc", "Current Acc", "Future Acc", "Past Acc"]
    for model in plot_models:
        table_data.append([rename_model(model), model_acc[model]["avg_acc"], model_acc[model]["current_acc"], model_acc[model]["future_acc"], model_acc[model]["past_acc"]])
    table_data = np.array(table_data)
    table_data = np.vstack((table_headers, table_data))
    table_data = np.array(table_data)
    # Save table to file
    np.savetxt(os.path.join("final_results", "static_results_table.csv"), table_data, delimiter=",", fmt="%s")