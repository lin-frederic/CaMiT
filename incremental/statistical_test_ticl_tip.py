import os
import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, shapiro
from sklearn.utils import shuffle
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

def statistical_test(heatmap1, heatmap2, n_permutations=10000):
    """
    Performs normality check, Wilcoxon signed-rank, permutation test, and t-test if applicable.
    """
    assert heatmap1.shape == heatmap2.shape
    differences = (heatmap1 - heatmap2).flatten()

    # 1. Normality Test
    stat, p_normality = shapiro(differences)
    is_normal = p_normality > 0.05
    print(f"Shapiro-Wilk normality test p-value: {p_normality} → {'Normal' if is_normal else 'Not normal'}")

    # 2. Wilcoxon Signed-Rank Test (non-parametric)
    w_stat, w_p = wilcoxon(differences)
    
    # 3. Permutation Test
    actual_diff = np.mean(differences)
    count = 0
    for _ in range(n_permutations):
        flipped = np.random.choice([-1, 1], size=differences.shape)
        permuted_diff = np.mean(differences * flipped)
        if abs(permuted_diff) >= abs(actual_diff):
            count += 1
    perm_p = count / n_permutations

    # 4. Paired t-test (if normal)
    if is_normal:
        t_stat, t_p = ttest_rel(heatmap1.flatten(), heatmap2.flatten())
        print(f"Paired t-test p-value: {t_p:}")
    else:
        t_p = None

    print(f"Wilcoxon p-value: {w_p}, Permutation p-value: {perm_p}")
    return {
        "shapiro_p": p_normality,
        "wilcoxon_p": w_p,
        "permutation_p": perm_p,
        "ttest_p": t_p,
    }

def plot_heatmap(matrix, title, xticklabels, yticklabels, mask_diag=True, cmap="viridis", annot=True):
    plt.figure(figsize=(10, 8))
    mask = np.eye(matrix.shape[0]) if mask_diag else None
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=yticklabels, annot=annot, cmap=cmap, cbar_kws={"label": "p-value"})
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    static_models = [
        "mocov3_s_lora",
    ]
    results_dirs = ["fecamv3_results", "randumb_results", "ranpac_results"]

    ticl_heatmaps = []
    ticl_labels = []
    for i, model in enumerate(static_models):
        for j, results_dir in enumerate(results_dirs):
            heatmap, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = results_to_heatmap(results_dir, model)
            ticl_heatmaps.append(heatmap)
            ticl_labels.append(f"{model} ({results_dir.split('_')[0]})")


    heatmap1, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="", model_name="car_mocov3_s_259")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap2, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="")
    print(avg_acc, future_acc, past_acc, current_acc)
    heatmap3, train_years, test_years, avg_acc, future_acc, past_acc, current_acc = tip_results(mode="replay")
    print(avg_acc, future_acc, past_acc, current_acc)

    tip_heatmaps = [heatmap1, heatmap2, heatmap3]
    tip_labels = ["MoCo v3-S +R", "MoCo v3-S +L$_c$", "MoCo v3-S +R+L$_a$"]

    wilcoxon_pvals = np.ones((len(ticl_heatmaps), len(tip_heatmaps)))
    permutation_pvals = np.ones((len(ticl_heatmaps), len(tip_heatmaps)))
    ttest_pvals = np.ones((len(ticl_heatmaps), len(tip_heatmaps)))
    shapiro_pvals = np.ones((len(ticl_heatmaps), len(tip_heatmaps)))

    for i in range(len(ticl_heatmaps)):
        for j in range(len(tip_heatmaps)):
            heatmap1 = ticl_heatmaps[i]
            heatmap2 = tip_heatmaps[j]
            stats = statistical_test(heatmap1, heatmap2)

            wilcoxon_pvals[i, j] = stats["wilcoxon_p"]
            permutation_pvals[i, j] = stats["permutation_p"]
            ttest_pvals[i, j] = stats["ttest_p"] if stats["ttest_p"] is not None else np.nan
            shapiro_pvals[i, j] = stats["shapiro_p"]
    plot_heatmap(wilcoxon_pvals, "Wilcoxon Signed-Rank Test p-values", ticl_labels, tip_labels)
    plot_heatmap(permutation_pvals, "Permutation Test p-values", ticl_labels, tip_labels)
    plot_heatmap(ttest_pvals, "Paired t-Test p-values", ticl_labels, tip_labels)
    plot_heatmap(shapiro_pvals, "Shapiro-Wilk Normality Test p-values (of differences)", ticl_labels, tip_labels)
