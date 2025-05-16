import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def results_to_heatmap(results_dir, model_name): # for ticl
    results_dir = os.path.join(results_dir, model_name)
    group_results = {}
    # Load JSON files
    for filename in os.listdir(results_dir):
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

        for filename in os.listdir(results_dir_model):
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



if __name__ == "__main__":
    heatmap1, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="", model_name="car_mocov3_s_259")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap2, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap3, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="replay")
    print(avg_acc, future_acc, past_acc, current_acc)

    heatmap4, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap("fecamv3_results", "mocov3_s_lora")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap5, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap("randumb_results", "mocov3_s_lora")
    heatmap6, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap("ranpac_results",  "mocov3_s_lora")
    print(avg_acc, future_acc, past_acc, current_acc)

    subplot_size = 6
    fig, axes = plt.subplots(1, 6, figsize=(subplot_size*6, subplot_size))


    vmin = 35
    vmax = 100
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
    axes[0].set_title(f"MoCo v3\n +R", fontsize=46)
    axes[0].set_ylabel("Train Year", fontsize=46)   
    ytick = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 8 == 0]
    ytick_labels = [x[2:] for x in train_years if (int(x) - 2007) % 8 == 0]
    axes[0].set_yticks(ytick)
    axes[0].set_yticklabels(ytick_labels, fontsize=46)
    axes[0].set_xlabel("Test Year", fontsize=46)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 8 == 0]
    xtick_labels = [x[2:] for x in test_years if (int(x) - 2007) % 8 == 0]
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
    axes[1].set_title(f"MoCo v3\n + L$_c$", fontsize=46)
    #axes[1].set_ylabel("Train Year", fontsize=46)
    # no yticks after the first plot
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', which='both', left=False, right=False)
    axes[1].set_xlabel("Test Year", fontsize=46)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 8 == 0]
    xtick_labels = [x[2:] for x in test_years if (int(x) - 2007) % 8 == 0]
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
    axes[2].set_title(f"MoCo v3\n + R + L$_a$", fontsize=46)
    # axes[2].set_ylabel("Train Year", fontsize=46)
    # no yticks after the first plot
    axes[2].set_yticks([])
    axes[2].set_yticklabels([])
    axes[2].tick_params(axis='y', which='both', left=False, right=False)
    axes[2].set_xlabel("Test Year", fontsize=46)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 8 == 0]
    xtick_labels = [x[2:] for x in test_years if (int(x) - 2007) % 8 == 0]
    axes[2].set_xticks(xtick)
    axes[2].set_xticklabels(xtick_labels, fontsize=46)
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False)

    sns.heatmap(
        heatmap4,
        ax=axes[3],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,
    )

    axes[3].set_title(f"MoCo v3\n +L$_i$ + FeCAM", fontsize=46)
    axes[3].set_yticks([])
    axes[3].set_yticklabels([])
    axes[3].tick_params(axis='y', which='both', left=False, right=False)
    axes[3].set_xlabel("Test Year", fontsize=46)
    xtick = [i + 0.5 for i, x in enumerate(test_years) if (int(x) - 2007) % 8 == 0]
    xtick_labels = [x[2:] for x in test_years if (int(x) - 2007) % 8 == 0]
    axes[3].set_xticks(xtick)
    axes[3].set_xticklabels(xtick_labels, fontsize=46)
    axes[3].tick_params(axis='x', which='both', bottom=False, top=False)

    sns.heatmap(
        heatmap5,
        ax=axes[4],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,
    )
    axes[4].set_title(f"MoCo v3\n +L$_i$ + RanDumb", fontsize=46)
    axes[4].set_yticks([])
    axes[4].set_yticklabels([])
    axes[4].tick_params(axis='y', which='both', left=False, right=False)
    axes[4].set_xlabel("Test Year", fontsize=46)
    axes[4].set_xticks(xtick)
    axes[4].set_xticklabels(xtick_labels, fontsize=46)
    axes[4].tick_params(axis='x', which='both', bottom=False, top=False)

    sns.heatmap(
        heatmap6,
        ax=axes[5],
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=False,  # Show colorbar in the last subplot
        cbar_kws={"shrink": 0.6, "label": "Accuracy (%)"}
    )
    axes[5].set_title(f"MoCo v3\n +L$_i$ + RanPAC", fontsize=46)
    axes[5].set_yticks([])
    axes[5].set_yticklabels([])
    axes[5].tick_params(axis='y', which='both', left=False, right=False)
    axes[5].set_xlabel("Test Year", fontsize=46)
    axes[5].set_xticks(xtick)
    axes[5].set_xticklabels(xtick_labels, fontsize=46)
    axes[5].tick_params(axis='x', which='both', bottom=False, top=False)

    # set aspect equal for all subplots
    for ax in axes:
        ax.set_aspect("equal")

    # add vertical line to separate the first 3 plots from the last 3
    line_position = 0.52
    fig.add_artist(plt.Line2D((line_position, line_position), (-0.1, 1.), color='black', lw=4, transform=fig.transFigure, figure=fig))
    fig.text(0.27, -0.15, "(a) Time-incremental pretraining", fontsize=46, ha="center")
    fig.text(0.76, -0.15, "(b) Time-incremental classifier learning", fontsize=46, ha="center")

    # plt.subplots_adjust(wspace=0.00, hspace=0.1, right=0.8, left=0.05,bottom=0.05, top=0.95)  # Shrink plot area to leave space for colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Add colorbar to the right of the last subplot
    cbar_ax = fig.add_axes([1., 0.22, 0.01, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=46)
    cbar.ax.set_ylabel("Accuracy", fontsize=46, va='center', labelpad=25)


    plt.tight_layout()
    plt.savefig("final_results/ti_results_heatmap.png", bbox_inches='tight')
    plt.show()


    
