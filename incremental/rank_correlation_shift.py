import os
import json
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr  # Handles >2 rankings as well

if __name__ == "__main__":
    models = ["car_mocov3_s_199", "car_mocov3_b_199", "clip_b"]
    
    label_scores = {}

    # Load scores for each model and label
    for model in models:
        results_dir = f"embedding_shift_results/{model}"
        label_files = os.listdir(results_dir)
        label_scores[model] = {}
        for label_file in label_files:
            with open(os.path.join(results_dir, label_file), "r") as f:
                data = json.load(f)
            label_scores[model][label_file] = []
            for train_year in data.keys():
                for train_yearB in data[train_year].keys():
                    label_scores[model][label_file].append(data[train_year][train_yearB])

    # Compute mean scores and rankings per model
    label_scores_dict = {}
    for model in models:
        label_scores_dict[model] = {}
        for k, v in label_scores[model].items():
            label_scores_dict[model][k] = float(np.mean(v))
        # Sort labels by mean score (descending)
        label_scores_dict[model] = sorted(label_scores_dict[model].items(), key=lambda x: x[1], reverse=True)

    # Build label-to-rank maps
    label_to_rank = {}
    for model in models:
        rankings = [item[0] for item in label_scores_dict[model]]
        label_to_rank[model] = {label: rank for rank, label in enumerate(rankings)}

    # Ensure all models have the same set of labels
    common_labels = set.intersection(*[set(label_to_rank[model].keys()) for model in models])
    if any(len(label_to_rank[model]) != len(common_labels) for model in models):
        raise ValueError("Not all models have the same set of labels.")

    # Create rank matrix: shape (num_models, num_labels)
    rank_matrix = np.array([
        [label_to_rank[model][label] for label in sorted(common_labels)]
        for model in models
    ])

    # Compute Spearman correlation matrix
    corr_matrix, _ = spearmanr(rank_matrix, axis=1)
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        xticklabels=models,
        yticklabels=models,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0, vmax=1,
        square=True,
        cbar_kws={"label": "Spearman Rank Correlation"}
    )
    plt.title("Spearman Rank Correlation Matrix")
    plt.tight_layout()
    plt.savefig("spearman_rank_correlation_matrix.png")
    plt.show()
