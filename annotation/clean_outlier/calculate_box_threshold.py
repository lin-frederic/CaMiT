import os
import json
import argparse

if __name__ =="__main__":
    # count number of annotations per bin
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="default")
    args = parser.parse_args()

    ANNOTATION_FILE = f"annotations_{args.user}.json"
    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

    bin_counts = {}
    for filename, occluded in annotations.items():
        image_id, score = filename.split("_")
        score = float(score.replace(".jpg", ""))
        if score > 0.5:
            bin_index = 5
        else:
            bin_index = int(score*10)
        bin_counts[bin_index] = bin_counts.get(bin_index, 0) + 1
    
    # plot error distribution
    import matplotlib.pyplot as plt
    import numpy as np
    bins = np.arange(0, 6)
    plt.plot(bins, [bin_counts.get(bin_index, 0) for bin_index in bins])
    for x, y in zip(bins, [bin_counts.get(bin_index, 0) for bin_index in bins]):
        plt.text(x, y-5, f"{y}", ha="center", va="bottom")
    # set x-label to 0.0-0.5
    xticks = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(0, 6)]
    xticks[-1] = "0.5+"
    plt.xticks(bins, xticks)
    plt.xlabel("Bin Index")
    plt.ylabel("Number of Annotations")
    plt.title("Distribution of error per bin")
    plt.show()