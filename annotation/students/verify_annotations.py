import os
import json
import numpy as np
annotators = ["flin", "hammar", "adrian"]
good_annotations = {}
bad_annotations = {}

for annotator in annotators:
    good_annotations_path = f"validation_test_set/{annotator}/not_selected_{annotator}.json"
    bad_annotations_path = f"validation_test_set/{annotator}/selected_{annotator}.json"
    with open(good_annotations_path) as f:
        good_annotations_by_annotator = json.load(f)
    with open(bad_annotations_path) as f:
        bad_annotations_by_annotator = json.load(f)
    
    for year in good_annotations_by_annotator:
        if year not in good_annotations:
            good_annotations[year] = {}
        for class_name in good_annotations_by_annotator[year]:
            if class_name not in good_annotations[year]:
                good_annotations[year][class_name] = []
            good_annotations[year][class_name].extend(good_annotations_by_annotator[year][class_name]) 
            # {year: {class_name: [validation_test_set/images/class_name/{"model"|empty}_{image_idx}_{crop_idx}_{crop_model}_{cluster_idx}_from_{source_class}.jpg"}]}
    for year in bad_annotations_by_annotator:
        if year not in bad_annotations:
            bad_annotations[year] = {}
        for class_name in bad_annotations_by_annotator[year]:
            if class_name not in bad_annotations[year]:
                bad_annotations[year][class_name] = []
            bad_annotations[year][class_name].extend(bad_annotations_by_annotator[year][class_name])
            # {year: {class_name: [validation_test_set/images/class_name/{"model"|empty}_{image_idx}_{crop_idx}_{crop_model}_{cluster_idx}_from_{source_class}.jpg"]}} 


with open("outputs/annotation_scores.json") as f:
    annotation_scores = json.load(f)
    # {outputs/test_models/class_name/year/{"model"|empty}_{image_idx}_{crop_idx}_{crop_model}_{cluster_idx}_from_{source_class}.jpg: {"annotation_score": score, "top_preds": top-5 predictions}} 

def map_validation_test_set_to_test_folder(good_annotations, bad_annotations):
    good_test_paths = []
    for year in good_annotations:
        for class_name in good_annotations[year]:
            for image in good_annotations[year][class_name]:
                good_test_paths.append(f"outputs/test_models/{class_name}/{year}/{image.split('/')[-1]}")
    bad_test_paths = []
    for year in bad_annotations:
        for class_name in bad_annotations[year]:
            for image in bad_annotations[year][class_name]:
                bad_test_paths.append(f"outputs/test_models/{class_name}/{year}/{image.split('/')[-1]}")
    return good_test_paths, bad_test_paths
    
good_test_paths, bad_test_paths = map_validation_test_set_to_test_folder(good_annotations, bad_annotations)

# check top-1,2,3,4,5 accuracies on good test paths
top_k_accuracies = {k: 0 for k in range(1, 6)}
for test_path in good_test_paths:
    if test_path in annotation_scores:
        class_name = test_path.split("/")[-3]
        top_preds = annotation_scores[test_path]["top_preds"] # [(class_name, score) for class_name, score in top-5 predictions]
        for k in range(1, 6):
            if class_name in [class_name for class_name, score in top_preds[:k]]:
                top_k_accuracies[k] += 1
top_k_accuracies = {k: v/len(good_test_paths) for k, v in top_k_accuracies.items()}
print("Number of good test paths:", len(good_test_paths))
print("Top-k accuracies on good test paths:")
print(top_k_accuracies)

# check top-1,2,3,4,5 accuracies on bad test paths
top_k_accuracies = {k: 0 for k in range(1, 6)}
for test_path in bad_test_paths:
    if test_path in annotation_scores:
        class_name = test_path.split("/")[-3]
        top_preds = annotation_scores[test_path]["top_preds"] # [(class_name, score
        for k in range(1, 6):
            if class_name in [class_name for class_name, score in top_preds[:k]]:
                top_k_accuracies[k] += 1
top_k_accuracies = {k: v/len(bad_test_paths) for k, v in top_k_accuracies.items()}
print("Number of bad test paths:", len(bad_test_paths))
print("Top-k accuracies on bad test paths:")
print(top_k_accuracies)

# check annotation score distribution on good and bad test paths
good_test_annotation_scores = []
for test_path in good_test_paths:
    if test_path in annotation_scores:
        good_test_annotation_scores.append(annotation_scores[test_path]["annotation_score"])
bad_test_annotation_scores = []
for test_path in bad_test_paths:
    if test_path in annotation_scores:
        bad_test_annotation_scores.append(annotation_scores[test_path]["annotation_score"])

good_mean = np.mean(good_test_annotation_scores)
good_std = np.std(good_test_annotation_scores)
bad_mean = np.mean(bad_test_annotation_scores)
bad_std = np.std(bad_test_annotation_scores)
print(f"Good Test Paths - Mean: {good_mean}, Std: {good_std}")
print(f"Bad Test Paths - Mean: {bad_mean}, Std: {bad_std}")
import matplotlib.pyplot as plt
import numpy as np
plt.hist(good_test_annotation_scores, bins=20, alpha=0.5, label='good test paths')
plt.hist(bad_test_annotation_scores, bins=20, alpha=0.5, label='bad test paths')
# add mean and std
plt.axvline(good_mean, color="blue", linestyle="solid", label=f"Good Test Paths - Mean: {good_mean:.2f}")
plt.axvline(good_mean + good_std, color="blue", linestyle="dashed", label=f"Good Test Paths - Std: {good_std:.2f}")
plt.axvline(good_mean - good_std, color="blue", linestyle="dashed")
plt.axvline(bad_mean, color="orange", linestyle="solid", label=f"Bad Test Paths - Mean: {bad_mean:.2f}")
plt.axvline(bad_mean + bad_std, color="orange", linestyle="dashed", label=f"Bad Test Paths - Std: {bad_std:.2f}")
plt.axvline(bad_mean - bad_std, color="orange", linestyle="dashed")
plt.xlabel("Annotation Score")
plt.ylabel("Frequency")
plt.title("Annotation Score Distribution")
plt.legend(loc='upper right')
plt.show()


# compute auroc
from sklearn.metrics import roc_curve, roc_auc_score
y_true = [1]*len(good_test_annotation_scores) + [0]*len(bad_test_annotation_scores)
y_scores = good_test_annotation_scores + bad_test_annotation_scores
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_auc})')
plt.legend(loc='lower right')
plt.show()

# plot good test paths with low annotation scores
good_low_annotation_scores = [test_path for test_path in good_test_paths if test_path in annotation_scores and annotation_scores[test_path]["annotation_score"] < 0.1]
print("Number of good test paths with low annotation scores:", len(good_low_annotation_scores))
import random
for i in range(5):
    test_path = random.choice(good_low_annotation_scores)
    annotation_class_name = test_path.split("/")[-3]
    annotation_score = annotation_scores[test_path]["annotation_score"]
    top_preds = annotation_scores[test_path]["top_preds"]
    top_preds = ", ".join([f"{class_name}: {score:.2f}" for class_name, score in top_preds])
    img = plt.imread(test_path)
    plt.imshow(img)
    plt.title(f"Class: {annotation_class_name}, Annotation Score: {annotation_score:.2f}, \n Top Predictions: {top_preds}")
    plt.show()
bad_high_annotation_scores = [test_path for test_path in bad_test_paths if test_path in annotation_scores and annotation_scores[test_path]["annotation_score"] > 0.9]
print("Number of bad test paths with high annotation scores:", len(bad_high_annotation_scores))

import random
for i in range(5):
    test_path = random.choice(bad_high_annotation_scores)
    annotation_class_name = test_path.split("/")[-3]
    annotation_score = annotation_scores[test_path]["annotation_score"]
    top_preds = annotation_scores[test_path]["top_preds"]
    top_preds = ", ".join([f"{class_name}: {score:.2f}" for class_name, score in top_preds])
    img = plt.imread(test_path)
    plt.imshow(img)
    plt.title(f"Class: {annotation_class_name}, Annotation Score: {annotation_score:.2f}, \n Top Predictions: {top_preds}")
    plt.show()
    
                
    

