import os
import json
import argparse
from dataset import SupervisedTimeDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_labels(annotation):
    """
    Extract labels grouped by year from the annotations.
    """
    labels = {}
    for image_name, image_annotations in tqdm(annotation.items(), desc="Processing annotations"):
        time = image_annotations["time"]
        if time not in labels:
            labels[time] = []
        for box in image_annotations["boxes"]:
            if box["class"] in ["unknown", "citroen_ds"]:
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

    # Load training and testing years
    train_years = sorted(os.listdir(args.train_images_dir), key=lambda x: int(x))  # e.g., 2007-2023
    test_years = sorted(os.listdir(args.test_images_dir), key=lambda x: int(x))

    # Load annotations
    with open(args.train_annotations, 'r') as f:
        train_annotations = json.load(f)
    with open(args.test_annotations, 'r') as f:
        test_annotations = json.load(f)

    # Extract labels
    train_labels = get_labels(train_annotations)
    test_labels = get_labels(test_annotations)

    # Get top 5 and bottom 5 classes across all years
    class_counts = {}
    for year in train_years:
        for label in train_labels[year]:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_classes = [label for label, count in sorted_classes[:5]]
    bot_classes = [label for label, count in sorted_classes[-5:]]

    # get 5 middle classes
    total_classes = len(sorted_classes)
    middle_start = total_classes // 2 - 2
    middle_end = middle_start + 5
    middle_classes = [label for label, count in sorted_classes[middle_start:middle_end]]

    # Count occurrences of top/bottom classes for each year
    class_year_counts = {label: [0] * len(train_years) for label in top_classes + middle_classes + bot_classes}
    for i, year in enumerate(train_years):
        year_labels = train_labels[year]
        for cls in top_classes + middle_classes + bot_classes:
            class_year_counts[cls][i] = year_labels.count(cls)

    # Set Seaborn theme
    sns.set_theme(style="whitegrid", palette="Set2")  # Use a clean theme and color palette
    plt.rcParams.update({'font.size': 14})  # Increase font size globally

    # Plot top 5 classes
    plt.figure(figsize=(12, 8))
    for cls in top_classes:
        sns.lineplot(x=train_years, y=class_year_counts[cls], label=cls, marker='o', linewidth=2.5)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Top 5 Classes Count Over Years', fontsize=18, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title="Classes", title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines
    plt.tight_layout()
    plt.savefig("top_classes_over_time.png", bbox_inches="tight", dpi=300)  # Save with high resolution
    plt.show()

    # Plot middle 5 classes
    plt.figure(figsize=(12, 8))
    for cls in middle_classes:
        sns.lineplot(x=train_years, y=class_year_counts[cls], label=cls, marker='o', linewidth=2.5)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Middle 5 Classes Count Over Years', fontsize=18, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title="Classes", title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines
    plt.tight_layout()
    plt.savefig("middle_classes_over_time.png", bbox_inches="tight", dpi=300)  # Save with high resolution
    plt.show()

    # Plot bottom 5 classes
    plt.figure(figsize=(12, 8))
    for cls in bot_classes:
        sns.lineplot(x=train_years, y=class_year_counts[cls], label=cls, marker='o', linewidth=2.5)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Bottom 5 Classes Count Over Years', fontsize=18, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12, title="Classes", title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines
    plt.tight_layout()
    plt.savefig("bot_classes_over_time.png", bbox_inches="tight", dpi=300)  # Save with high resolution
    plt.show()