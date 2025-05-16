import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to train annotations JSON file", default="cars_dataset/train_annotations.json")
    parser.add_argument("--train_images_dir", type=str, help="Path to train images directory", default="cars_dataset/train_blurred")
    parser.add_argument("--test_annotations", type=str, help="Path to test annotations JSON file", default="cars_dataset/test_annotations.json")
    parser.add_argument("--test_images_dir", type=str, help="Path to test images directory", default="cars_dataset/test_blurred")
    args = parser.parse_args()

    test_years = sorted(os.listdir(args.test_images_dir), key=lambda x: int(x))
    with open(args.test_annotations, 'r') as f:
        test_annotations = json.load(f)

    test_labels = get_labels(test_annotations)

    # Initialize stats
    appearing_counts = []
    disappearing_counts = []
    existing_counts = []
    years = test_years

    prev_classes = set()
    future_classes_map = {}

    for i in range(len(test_years)):
        future_classes = set()
        for j in range(i, len(test_years)):
            future_classes.update(test_labels[test_years[j]])
        future_classes_map[test_years[i]] = future_classes

    for i, year in enumerate(years):
        current_classes = set(test_labels[year])
        future_classes = future_classes_map[year]

        appearing = current_classes - prev_classes
        existing = current_classes & prev_classes
        disappearing = prev_classes - future_classes if i > 0 else set()

        appearing_counts.append(len(appearing))
        existing_counts.append(len(existing))
        disappearing_counts.append(len(disappearing))

        prev_classes.update(current_classes)

    # Plotting
    x = [y[2:] for y in years]  # Shorten year labels
    width = 0.6

    plt.figure(figsize=(20, 17.8))

    # Plot Appearing classes
    plt.bar(x, appearing_counts, label="New", color="#33CC33", align='edge', width=width)
    # Plot Existing classes
    plt.bar(x, existing_counts, bottom=appearing_counts, label="Existing", color="#3386CC", align='edge', width=width)
    # Plot Disappearing classes with hatching pattern (rouge hachuré)
    bottom_sum = [a + e for a, e in zip(appearing_counts, existing_counts)]
    plt.bar(x, disappearing_counts, bottom=bottom_sum, label="Removed", color="#FF3333", hatch='///', align='edge', width=width)

    # Filter for odd years only
    odd_years = [year[2:] for year in years if (int(year) -2007) % 8 == 0]

    # Set custom x-ticks for odd years
    odd_years_indices = [i+0.3 for i, year in enumerate(years) if (int(year) -2007) % 8 == 0]
    plt.gca().tick_params(axis='x', pad=12)
    plt.xticks(odd_years_indices, odd_years, fontsize=60)
    plt.yticks(fontsize=60)

    # Set labels and title with larger font size
    plt.xlabel("Year", fontsize=60)
    plt.ylabel("Number of Classes", fontsize=60)

    # Adjust legend
    plt.legend(fontsize=60, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)

    # Set explicit x-axis limits with a small buffer
    plt.xlim(-0.3 , len(years)-0.15)  # Add a small buffer on both sides

    # set aspect ratio
    plt.gca().set_aspect('auto', adjustable='box')

    # Adjust layout
    plt.tight_layout()
    plt.savefig("final_results/class_dynamics_histogram.png", bbox_inches="tight")
    plt.show()
