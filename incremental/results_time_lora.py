import os
import json
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


if __name__ == "__main__":
    results_dir = "ncm_results"

    # List all models and filter by the specific model name
    
    model_name = "mocov3_s_lora"
    years = [str(x) for x in range(2008, 2024)]
    year_models = [f"{model_name}_{year}" for year in years]
    years.insert(0, "2007")
    year_models.insert(0, model_name)

    # Initialize dictionaries to store accuracies
    avg_acc = {}
    avg_current_acc = {}
    avg_future_acc = {}
    avg_past_acc = {}

    # Process each model
    for model in year_models:
        heatmap_data, train_years, test_years, avg_acc_model, future_acc_model, past_acc_model, current_acc_model = results_to_heatmap(results_dir, model) 
        avg_acc[model] = avg_acc_model
        avg_future_acc[model] = future_acc_model
        avg_past_acc[model] = past_acc_model
        avg_current_acc[model] = current_acc_model

    # Extract years from model names
    years = [int(model.split("_")[-1]) if model.split("_")[-1].isdigit() else 2007 for model in year_models]


    # Convert dictionaries to lists for plotting
    avg_acc_values = [avg_acc[model] for model in year_models]
    avg_future_acc_values = [avg_future_acc[model] for model in year_models]
    avg_past_acc_values = [avg_past_acc[model] for model in year_models]
    avg_current_acc_values = [avg_current_acc[model] for model in year_models]

    # Prepare data for Seaborn
    data = {
        "Year": years * 4,  # Repeat years for each accuracy type
        "Accuracy": avg_acc_values + avg_current_acc_values +  avg_future_acc_values + avg_past_acc_values,
        "Type": ["Overall"] * len(years) + ["Current"]*len(years) + ["Future"] * len(years) + ["Past"] * len(years),
    }
    # Create a DataFrame for Seaborn
    import pandas as pd
    df = pd.DataFrame(data)

    # Plotting with Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Year",
        y="Accuracy",
        hue="Type",
        style="Type",
        markers=True,
        dashes=False,
        palette="Set1",
    )

    # Add labels and title
    plt.title("Evolution of Average Accuracies Over Years", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(title="Accuracy Type", fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig("average_accuracies_over_years.png")
    plt.show()