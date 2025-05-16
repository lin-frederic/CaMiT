import os
import json
import argparse
from dataset import SupervisedTimeDataset
from tqdm import tqdm


def get_labels(annotation):
    labels = {}
    for image_name, image_annotations in tqdm(annotation.items()):
        time = image_annotations["time"]
        if time not in labels:
            labels[time] = []
        for box in image_annotations["boxes"]:
            if box["class"] == "unknown":
                continue
            if box["class"] == "citroen_ds":
                continue
            labels[time].append(box["class"])
    return labels

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to train annotations JSON file", default="cars_dataset/train_annotations.json")
    parser.add_argument("--train_images_dir", type=str, help="Path to train images directory", default="cars_dataset/train_blurred")
    parser.add_argument("--test_annotations", type=str, help="Path to test annotations JSON file", default="cars_dataset/test_annotations.json")
    parser.add_argument("--test_images_dir", type=str, help="Path to test images directory", default="cars_dataset/test_blurred")
    args = parser.parse_args()

    train_years = sorted(os.listdir(args.train_images_dir), key=lambda x: int(x)) # 2007-2023: 17 years
    test_years = sorted(os.listdir(args.test_images_dir), key=lambda x: int(x))

    imbalance = {}
    threshold = 0

    with open(args.train_annotations, 'r') as f:
        train_annotations = json.load(f)
    
    with open(args.test_annotations, 'r') as f:
        test_annotations = json.load(f)

    train_labels = get_labels(train_annotations)
    test_labels = get_labels(test_annotations)

    with tqdm(total=len(train_years)*len(test_years)) as pbar:
        for train_year in train_years:
            labels_count = {}
            for label in train_labels[train_year]:
                if label not in labels_count:
                    labels_count[label] = 0
                labels_count[label] += 1
            train_labels_time = [label for label in train_labels[train_year] if labels_count[label] > threshold]
            train_labels_time = set(train_labels_time)
            imbalance[train_year] = {}
            for test_year in test_years:
                test_labels_time = set(test_labels[test_year])
 
                # compute |A ∩ B| / |B| (checking how many labels in train are in test)
                intersection = len(train_labels_time.intersection(test_labels_time))
                union = len(test_labels_time)
                jaccard_similarity = intersection / union if union != 0 else 0 # modified jaccard similarity
                imbalance[train_year][test_year] = jaccard_similarity*100
                pbar.update(1)
    
    # plot matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    train_years = sorted(imbalance.keys())
    test_years = sorted(imbalance[train_years[0]].keys())
    imbalance_matrix = np.zeros((len(train_years), len(test_years)))
    for i, train_year in enumerate(train_years):
        for j, test_year in enumerate(test_years):
            imbalance_matrix[i][j] = imbalance[train_year][test_year]
    test_years = [x[2:] for x in test_years]
    train_years = [x[2:] for x in train_years]
    plt.figure(figsize=(10, 8))
    sns.heatmap(imbalance_matrix, cmap="viridis", xticklabels=test_years, yticklabels=train_years)
    plt.xlabel("Test Year")
    plt.ylabel("Train Year")
    plt.tight_layout()
    plt.savefig("final_results/imbalance_matrix.png", bbox_inches='tight')
    plt.show()