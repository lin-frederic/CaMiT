import os
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter  # Import ScalarFormatter for formatting
from matplotlib import gridspec


if __name__ == "__main__":
    results_dir = "embedding_shift_results/clip_b"
    label_files = os.listdir(results_dir)
    embedding_shift = {}
    label_scores = {}
    for label_file in label_files:
        with open(os.path.join(results_dir, label_file), "r") as f:
            data = json.load(f)
        label_scores[label_file] = []
        for train_year in data.keys():
            if train_year not in embedding_shift:
                embedding_shift[train_year] = {}
            for train_yearB in data[train_year].keys():
                if train_yearB not in embedding_shift[train_year]:
                    embedding_shift[train_year][train_yearB] = []
                embedding_shift[train_year][train_yearB].append(data[train_year][train_yearB])
                label_scores[label_file].append(data[train_year][train_yearB])
    

    # Compute average
    avg_embedding_shift = {}
    for train_year in embedding_shift.keys():
        avg_embedding_shift[train_year] = {}
        for train_yearB in embedding_shift[train_year].keys():
            avg_embedding_shift[train_year][train_yearB] = np.mean(embedding_shift[train_year][train_yearB])

    heatmap_data = np.zeros((len(avg_embedding_shift), len(avg_embedding_shift)))
    train_years = sorted(avg_embedding_shift.keys(), key=lambda x: int(x))
    train_yearsB = sorted(avg_embedding_shift[train_years[0]].keys(), key=lambda x: int(x))
    for i, train_year in enumerate(train_years):
        for j, train_yearB in enumerate(train_yearsB):
            heatmap_data[i, j] = avg_embedding_shift[train_year][train_yearB]

    # save the heatmap data with numpy
    np.save("final_results/avg_embedding_shift.npy", heatmap_data)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)

    # Main heatmap axis (left, bottom, width, height) in figure coordinates
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])

    # Colorbar axis – adjust height here (smaller height to reduce size)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.75])  # [left, bottom, width, height]
    sns.heatmap(
        heatmap_data,
        xticklabels=False,
        yticklabels=False,
        annot=False,
        fmt=".1f",
        cmap="Spectral_r",
        ax=ax,
        cbar_ax=cbar_ax
    )

    ax.set_aspect("equal")

    # Set ticks
    xticks = [i + 0.5 for i, x in enumerate(train_yearsB) if (int(x) - 2007) % 8 == 0]
    xticklabels = [x[2:] for x in train_yearsB if (int(x) - 2007) % 8 == 0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=50)

    yticks = [i + 0.5 for i, x in enumerate(train_years) if (int(x) - 2007) % 8 == 0]
    yticklabels = [x[2:] for x in train_years if (int(x) - 2007) % 8 == 0]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=50)

    # Format colorbar
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    cbar_ax.yaxis.set_major_formatter(formatter)
    cbar_ax.tick_params(labelsize=50)
    cbar_ax.yaxis.get_offset_text().set(size=50)
    cbar_ax.yaxis.offsetText.set_x(10)
    cbar_ax.set_ylabel("KID", fontsize=50, labelpad=20)

    ax.set_xlabel("Year", fontsize=50)
    ax.set_ylabel("Year", fontsize=50)

    plt.savefig("final_results/heatmap_embedding_shift.png", bbox_inches='tight')
    plt.show()


    # for each label, pool the scores
    label_scores_dict = {k: float(np.mean(v)) for k, v in label_scores.items() if len(v) == 17*17}
    label_scores = sorted(label_scores_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top5 and bottom5 labels
    top_labels = label_scores[:5]
    bot_labels = label_scores[-5:]
    top_labels = [x[0] for x in top_labels]
    bot_labels = [x[0] for x in bot_labels]

    # Compute global min and max values for consistent color scale
    all_heatmap_data = []
    for label_file in top_labels + bot_labels:
        with open(os.path.join(results_dir, label_file), "r") as f:
            data = json.load(f)
        avg_embedding_shift = {}
        for train_year in data.keys():
            avg_embedding_shift[train_year] = {}
            for train_yearB in data[train_year].keys():
                avg_embedding_shift[train_year][train_yearB] = np.mean(data[train_year][train_yearB])
        heatmap_data = np.zeros((len(avg_embedding_shift), len(avg_embedding_shift)))
        train_years = sorted(avg_embedding_shift.keys(), key=lambda x: int(x))
        train_yearsB = sorted(avg_embedding_shift[train_years[0]].keys(), key=lambda x: int(x))
        for j, train_year in enumerate(train_years):
            for k, train_yearB in enumerate(train_yearsB):
                heatmap_data[j, k] = avg_embedding_shift[train_year][train_yearB]
        all_heatmap_data.append(heatmap_data)

    global_min = np.min([np.min(data) for data in all_heatmap_data])
    global_max = np.max([np.max(data) for data in all_heatmap_data])

    # Create subplots to compare top varying classes with bottom varying classes
    subplot_size = 6
    fig, axes = plt.subplots(2,5, figsize=(subplot_size * 5, subplot_size * 2))

    # Plot top5 labels
    for i, label_file in enumerate(top_labels):
        with open(os.path.join(results_dir, label_file), "r") as f:
            data = json.load(f)
        avg_embedding_shift = {}
        for train_year in data.keys():
            avg_embedding_shift[train_year] = {}
            for train_yearB in data[train_year].keys():
                avg_embedding_shift[train_year][train_yearB] = np.mean(data[train_year][train_yearB])
        heatmap_data = np.zeros((len(avg_embedding_shift), len(avg_embedding_shift)))
        train_years = sorted(avg_embedding_shift.keys(), key=lambda x: int(x))
        train_yearsB = sorted(avg_embedding_shift[train_years[0]].keys(), key=lambda x: int(x))
        for j, train_year in enumerate(train_years):
            for k, train_yearB in enumerate(train_yearsB):
                heatmap_data[j, k] = avg_embedding_shift[train_year][train_yearB]

        axes[0, i].imshow(heatmap_data, cmap="Spectral_r", aspect="equal", vmin=global_min, vmax=global_max)
        if i == 0:
            axes[0, i].set_yticks([i for i, x in enumerate(train_years) if (int(x) - 2007) % 8 == 0])
            axes[0, i].set_yticklabels([x[2:] for x in train_years if (int(x) - 2007) % 8 == 0], fontsize=50)
            axes[0, i].set_ylabel("Year", fontsize=50)
        else:
            axes[0, i].set_yticks([])
            axes[0, i].set_yticklabels([])
            axes[0, i].tick_params(axis='y', which='both', left=False, right=False)
        
        axes[0, i].set_xticks([])
        axes[0, i].set_xticklabels([])
        axes[0, i].tick_params(axis='x', which='both', bottom=False, top=False)
        label = label_file.replace("results_", "").replace(".json", "")
        label = label.replace("_", "\n").title()
        label = label.replace("Xc90", "XC90")
        axes[0, i].set_title(label, fontsize=50)

    # Plot bottom5 labels
    for i, label_file in enumerate(bot_labels):
        with open(os.path.join(results_dir, label_file), "r") as f:
            data = json.load(f)
        avg_embedding_shift = {}
        for train_year in data.keys():
            avg_embedding_shift[train_year] = {}
            for train_yearB in data[train_year].keys():
                avg_embedding_shift[train_year][train_yearB] = np.mean(data[train_year][train_yearB])
        heatmap_data = np.zeros((len(avg_embedding_shift), len(avg_embedding_shift)))
        train_years = sorted(avg_embedding_shift.keys(), key=lambda x: int(x))
        train_yearsB = sorted(avg_embedding_shift[train_years[0]].keys(), key=lambda x: int(x))
        for j, train_year in enumerate(train_years):
            for k, train_yearB in enumerate(train_yearsB):
                heatmap_data[j, k] = avg_embedding_shift[train_year][train_yearB]

        axes[1, i].imshow(heatmap_data, cmap="Spectral_r", aspect="equal", vmin=global_min, vmax=global_max)
        if i == 0:
            axes[1, i].set_yticks([i for i, x in enumerate(train_years) if (int(x) - 2007) % 8 == 0])
            axes[1, i].set_yticklabels([x[2:] for x in train_years if (int(x) - 2007) % 8 == 0], fontsize=50)
            axes[1, i].set_ylabel("Year", fontsize=50)
        else:
            axes[1, i].set_yticks([])
            axes[1, i].set_yticklabels([])
            axes[1, i].tick_params(axis='y', which='both', left=False, right=False)
        axes[1, i].set_xticks([i for i, x in enumerate(train_yearsB) if (int(x) - 2007) % 8 == 0])
        axes[1, i].set_xticklabels([x[2:] for x in train_yearsB if (int(x) - 2007) % 8 == 0], fontsize=50)
        axes[1, i].set_xlabel("Year", fontsize=50)
        label = label_file.replace("results_", "").replace(".json", "")
        label = label.replace("_", "\n").title()
        label = label.replace("2Cv", "2CV")
        axes[1, i].set_title(label, fontsize=50)
    
    # Add colorbar using a proxy ScalarMappable
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Define the position and size of the color bar
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=plt.Normalize(vmin=global_min, vmax=global_max))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=50)  # Set color bar tick label size
    cbar.ax.yaxis.get_offset_text().set(size=50)  # Optional: match the scientific offset font size
    cbar.ax.yaxis.offsetText.set_x(3.3)  # Adjust the position of the offset text
    
    # Use ScalarFormatter with scientific notation
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))  # Force scientific notation
    # add label to color bar
    cbar.set_label("KID", fontsize=50, labelpad=20)

    #fig.subplots_adjust(left=0.03, right=0.90, top=0.95, bottom=0.05)
    fig.subplots_adjust(hspace=0.4)

    plt.savefig("final_results/heatmap_embedding_shift_top_bottom.png", bbox_inches='tight')
    plt.show()