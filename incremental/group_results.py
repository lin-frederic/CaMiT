import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Group results")
    parser.add_argument("--results_dir", type=str, help="Path to results directory", default="ncm_results")
    parser.add_argument("--model", type=str, help="Model name", default="dinov2_s")

    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, args.model)
    method_name = os.path.basename(args.results_dir).split("_results")[0]
    model = args.model
    group_results = {}
    for filename in tqdm(os.listdir(results_dir), desc="Processing files"):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r") as f:
                data = json.load(f)
            # Process the data
            # For example, you can print the keys and values
            group_results = {**group_results, **data}  # Merge dictionaries

    # ncm_results: {train_year: {test_year: acc}}
    # plot the results
    train_years = sorted(group_results.keys(), key=lambda x: int(x))
    test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))

    heatmap_data = np.zeros((len(train_years), len(test_years)))
    avg_acc = 0
    current_acc = 0

    for i, train_year in enumerate(train_years):
        for j, test_year in enumerate(test_years):
            heatmap_data[i, j] = round(group_results[train_year][test_year]*100, 2)
            avg_acc += group_results[train_year][test_year]
            if i == j:
                current_acc += group_results[train_year][test_year]
    avg_acc /= (len(train_years) * len(test_years))
    avg_acc = round(avg_acc * 100, 2)
    current_acc = round(current_acc * 100 / len(train_years), 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=test_years, yticklabels=train_years, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"{method_name} - {model} - Average Accuracy: {avg_acc:.2f}% - Current Accuracy: {current_acc:.2f}%")
    plt.xlabel("Test Year")
    plt.ylabel("Train Year")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"heatmap_{args.model}.png"))
    plt.show()