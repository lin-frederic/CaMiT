import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def tip_results(mode, model_name="mocov3_s_lora"):
    years = [str(x) for x in range(2008, 2024)]
    year_models = [f"{model_name}_{year}_{mode}" if mode else f"{model_name}_{year}" for year in years]
    years.insert(0, "2007")
    if model_name == "mocov3_s_lora":
        year_models.insert(0, model_name)
    elif model_name == "car_mocov3_s_259":
        year_models.insert(0, "car_mocov3_s_199")

    results_over_time = np.zeros((17, 17))

    for i, model in enumerate(year_models):
        results_dir_model = os.path.join("ncm_results", model)
        group_results = {}

        for filename in tqdm(os.listdir(results_dir_model), desc=f"Processing {model}"):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir_model, filename), "r") as f:
                    data = json.load(f)
                group_results = {**group_results, **data}
        train_years = sorted(group_results.keys(), key=lambda x: int(x))
        test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))
        model_year = train_years[i]
        model_results = group_results[model_year]
        for j, test_year in enumerate(model_results.keys()):
            results_over_time[i, j] = round(model_results[test_year] * 100, 2)
    
    avg_acc = 0
    past_acc = 0
    future_acc = 0
    current_acc = 0

    for i in range(len(year_models)):
        for j in range(len(test_years)):
            avg_acc += results_over_time[i, j]
            if i < j:
                future_acc += results_over_time[i, j]
            elif i > j:
                past_acc += results_over_time[i, j]
            else:
                current_acc += results_over_time[i, j]
    avg_acc /= (len(year_models) * len(test_years))
    off_diagonal_count = len(year_models) * (len(year_models) - 1)/2
    future_acc /= off_diagonal_count
    past_acc /= off_diagonal_count
    current_acc /= len(year_models)
    return results_over_time, train_years, test_years, avg_acc, future_acc, past_acc, current_acc


def rename_model(model_name):
    if "dinov2" in model_name:
        model_size = model_name.split("_")[-1]
        model_name = f"DinoV2-{model_size}"
    elif "clip" in model_name:
        model_size = model_name.split("_")[1]
        if "lora" in model_name:
            model_name = f"CLIP-{model_size} + L"
        else:
            model_name = f"CLIP-{model_size}"
    elif "car_mocov3" in model_name:
        model_size = model_name.split("_")[2]
        model_name = f"MoCoV3-{model_size}"
    elif "mocov3" in model_name:
        model_size = model_name.split("_")[1]
        model_name = f"MoCoV3-{model_size} + L"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_name

if __name__ == "__main__":

    heatmap1, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="", model_name="car_mocov3_s_259")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap2, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap3, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="replay")
    print(avg_acc, future_acc, past_acc, current_acc)
    
    #vmin = min(heatmap1.min(), heatmap2.min(), heatmap3.min())
    #vmax = max(heatmap1.max(), heatmap2.max(), heatmap3.max())
    vmin = 35
    vmax = 100
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    sns.heatmap(
        heatmap1,
        ax=axes[0],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,
    )
    axes[0].set_title(f"MoCoV3-S +R", fontsize=46)
    axes[0].set_ylabel("Train Year", fontsize=38)   
    ytick = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 4 == 0]
    ytick_labels = [x for x in train_years if (int(x) - 2007) % 4 == 0]
    axes[0].set_yticks(ytick)
    axes[0].set_yticklabels(ytick_labels, fontsize=46)
    axes[0].set_xlabel("Test Year", fontsize=38)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 4 == 0]
    xtick_labels = [x for x in test_years if (int(x) - 2007) % 4 == 0]
    axes[0].set_xticks(xtick)
    axes[0].set_xticklabels(xtick_labels, fontsize=46)
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[0].tick_params(axis='y', which='both', left=False, right=False)
    sns.heatmap(
        heatmap2,
        ax=axes[1],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,
    )
    axes[1].set_title(f"MoCoV3-S + Lc", fontsize=46)
    #axes[1].set_ylabel("Train Year", fontsize=38)
    # no yticks after the first plot
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', which='both', left=False, right=False)
    axes[1].set_xlabel("Test Year", fontsize=38)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 4 == 0]
    xtick_labels = [x for x in test_years if (int(x) - 2007) % 4 == 0]
    axes[1].set_xticks(xtick)
    axes[1].set_xticklabels(xtick_labels, fontsize=46)
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False)
    sns.heatmap(
        heatmap3,
        ax=axes[2],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,
    )
    axes[2].set_title(f"MoCoV3-S + R + La", fontsize=46)
    # axes[2].set_ylabel("Train Year", fontsize=38)
    # no yticks after the first plot
    axes[2].set_yticks([])
    axes[2].set_yticklabels([])
    axes[2].tick_params(axis='y', which='both', left=False, right=False)
    axes[2].set_xlabel("Test Year", fontsize=38)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 4 == 0]
    xtick_labels = [x for x in test_years if (int(x) - 2007) % 4 == 0]
    axes[2].set_xticks(xtick)
    axes[2].set_xticklabels(xtick_labels, fontsize=46)
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False)
    # Adjust layout to make room for the colorbar
    # plt.subplots_adjust(wspace=0.00, hspace=0.1, right=0.8, left=0.05,bottom=0.05, top=0.95)  # Shrink plot area to leave space for colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Add colorbar to the right of the last subplot
    cbar_ax = fig.add_axes([1., 0.16, 0.02, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=46)
    cbar.set_label("Accuracy (%)", fontsize=46, labelpad=20)
    plt.tight_layout()
    plt.savefig(os.path.join("final_results", "tip_results_heatmap.png"), bbox_inches='tight')
    plt.show()    