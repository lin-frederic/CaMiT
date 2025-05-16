import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Group results")
    parser.add_argument("--results_dir", type=str, help="Path to results directory", default="ranpac_results")
    parser.add_argument("--model", type=str, help="Model name", default="clip_b_lora")

    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, args.model)
    method_name = os.path.basename(args.results_dir).split("_results")[0]
    model = args.model
    group_results = {}
    for filename in tqdm(os.listdir(results_dir), desc="Processing files"):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r") as f:
                data = json.load(f)
            for train_year in data.keys():
                if train_year not in group_results:
                    group_results[train_year] = {}
                for test_year in data[train_year].keys():
                    if test_year not in group_results[train_year]:
                        group_results[train_year][test_year] = []
                    group_results[train_year][test_year].append(data[train_year][test_year])
    


    # ncm_results: {train_year: {test_year: acc}}
    # plot the results
    train_years = sorted(group_results.keys(), key=lambda x: int(x))
    test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))

    heatmap_data = np.zeros((len(train_years), len(test_years)))
    stdev_data = np.zeros((len(train_years), len(test_years)))
    avg_acc = [0 for _ in range(len(group_results[train_years[0]][test_years[0]]))]

    for i, train_year in enumerate(train_years):
        for j, test_year in enumerate(test_years):
            heatmap_data[i, j] = round(np.mean(group_results[train_year][test_year])*100, 2)
            stdev_data[i, j] = round(np.std(group_results[train_year][test_year])*100, 2)
            for k in range(len(group_results[train_year][test_year])):
                avg_acc[k] += group_results[train_year][test_year][k]
    
    avg_acc = [x / (len(train_years) * len(test_years)) for x in avg_acc]
    avg_acc_std = np.std(avg_acc)
    avg_acc = np.mean(avg_acc)
    
    avg_acc = round(avg_acc * 100, 2)
    avg_acc_std = round(avg_acc_std * 100, 2)

    
    # Create custom annotation strings with mean ± std
    annot_data = np.empty_like(heatmap_data, dtype=object)
    for i in range(len(train_years)):
        for j in range(len(test_years)):
            mean = heatmap_data[i, j]
            std = stdev_data[i, j]
            annot_data[i, j] = f"{mean:.1f}±{std:.1f}"

    # Draw heatmap with custom annotations
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, xticklabels=test_years, yticklabels=train_years,
                annot=annot_data, fmt="", cmap="viridis")


    
    plt.title(f"{method_name} - {model} - Average Accuracy: {avg_acc:.2f} ± {avg_acc_std:.2f}%")
    plt.xlabel("Test Year")
    plt.ylabel("Train Year")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"heatmap_{args.model}.png"))
    plt.show()