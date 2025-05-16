import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    results_dir = "ncm_results"

    # List all models and filter by the specific model name
    model_name = "clip_b_lora"
    years = [str(x) for x in range(2008, 2024)]
    year_models = [f"{model_name}_{year}" for year in years]
    years.insert(0, "2007")
    year_models.insert(0, model_name)

    # Process each model
    results_over_time = np.zeros((17, 17))
    initial_results = np.zeros((17, 17))


    for i,model in enumerate(year_models):
        results_dir_model = os.path.join("ncm_results", model)
        group_results = {}

        # Load JSON files
        for filename in tqdm(os.listdir(results_dir_model), desc=f"Processing {model}"):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir_model, filename), "r") as f:
                    data = json.load(f)
                group_results = {**group_results, **data}

        # Sort train and test years
        train_years = sorted(group_results.keys(), key=lambda x: int(x))
        test_years = sorted(group_results[train_years[0]].keys(), key=lambda x: int(x))
        model_year = train_years[i]
        model_results = group_results[model_year]
        for j, test_year in enumerate(model_results.keys()):
            results_over_time[i, j] = round(model_results[test_year] * 100, 2)
        if i == 0:
            for k, train_year in enumerate(train_years):
                for j, test_year in enumerate(test_years):
                    initial_results[k, j] = round(group_results[train_year][test_year] * 100, 2)

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

    print(f"Avg Acc: {avg_acc:.2f}%")
    print(f"Current Acc: {current_acc:.2f}%")
    print(f"Past Acc: {past_acc:.2f}%")
    print(f"Future Acc: {future_acc:.2f}%")

    plt.figure(figsize=(10, 8))
    year_models = [x.split("_")[-1] for x in year_models]
    results_over_time = results_over_time
    sns.heatmap(results_over_time, xticklabels=test_years, yticklabels=train_years, annot=False, fmt=".1f", cmap="viridis")
    plt.title(f"Results Over Time - {model_name} - {avg_acc:.2f}%")
    plt.xlabel("Test Year")
    plt.ylabel("Train Year")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"results_over_time_{model_name}.png"))
    plt.show()



        