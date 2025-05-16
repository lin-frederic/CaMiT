import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group results")
    parser.add_argument("--model", type=str, help="Model name", default="dinov2_s")

    args = parser.parse_args()

    methods = ["ncm", "fecam_common_v3", "fecamv3", "ranpac", "randumb"]

    # Step 1: Collect all accuracy data to determine global min and max
    all_heatmap_data = []

    for method in methods:
        results_dir = os.path.join(f"{method}_results", args.model)
        group_results = {}
        for filename in tqdm(os.listdir(results_dir), desc=f"Processing files for {method}"):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    data = json.load(f)
                # Merge the data into group_results
                group_results = {**group_results, **data}

        train_years = sorted(group_results.keys(), key=lambda x: int(x))
        test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))
        heatmap_data = np.zeros((len(train_years), len(test_years)))

        for i, train_year in enumerate(train_years):
            for j, test_year in enumerate(test_years):
                heatmap_data[i, j] = round(group_results[train_year][test_year] * 100, 2)

        # Append the heatmap data to the list for global normalization
        all_heatmap_data.append(heatmap_data)

    # Calculate global min and max values
    all_heatmap_data = np.concatenate(all_heatmap_data)  # Flatten all data into a single array
    global_min = np.min(all_heatmap_data)
    global_max = np.max(all_heatmap_data)

    # Step 2: Plot heatmaps with the same scale
    os.makedirs(f"final_results", exist_ok=True)
    os.makedirs(f"final_results/{args.model}", exist_ok=True)

    for method in methods:
        results_dir = os.path.join(f"{method}_results", args.model)
        group_results = {}
        for filename in tqdm(os.listdir(results_dir), desc=f"Processing files for {method}"):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    data = json.load(f)
                # Merge the data into group_results
                group_results = {**group_results, **data}

        train_years = sorted(group_results.keys(), key=lambda x: int(x))
        test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))
        heatmap_data = np.zeros((len(train_years), len(test_years)))
        avg_acc = 0

        for i, train_year in enumerate(train_years):
            for j, test_year in enumerate(test_years):
                heatmap_data[i, j] = round(group_results[train_year][test_year] * 100, 2)
                avg_acc += group_results[train_year][test_year]

        avg_acc /= (len(train_years) * len(test_years))
        avg_acc = round(avg_acc * 100, 2)

        # Plot the heatmap with the global scale
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            xticklabels=test_years,
            yticklabels=train_years,
            annot=False,
            fmt=".1f",
            cmap="viridis",
            vmin=global_min,  # Use global min
            vmax=global_max   # Use global max
        )
        if method == "fecam_common_v3":
            method_name = "fecam-1"
        elif method == "fecamv3":
            method_name = "fecam-N"
        elif method == "ncm":
            method_name = "NCM"
        elif method == "ranpac":
            method_name = "RanPAC"
        elif method == "randumb":
            method_name = "Randumb"
        
        if args.model == "car_mocov3_s_199":
            model_name = "mocov3_s"
        elif args.model == "car_mocov3_b_199":
            model_name = "mocov3_b"
        else:
            model_name = args.model
        plt.title(f"{method_name} - {model_name} - {avg_acc:.2f}%")
        plt.xlabel("Test Year")
        plt.ylabel("Train Year")
        plt.tight_layout()
        plt.savefig(os.path.join(f"final_results/{args.model}", f"heatmap_{method_name}.png"))
        plt.show()