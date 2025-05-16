import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_annotations", type=str, help="Path to the student annotations", default="outputs/student_annotations_clean.json")
    args = parser.parse_args()

    with open(args.student_annotations) as f:
        student_annotations = json.load(f)

    model_probability_distribution = {}
    for image_id, annotation in tqdm(student_annotations.items(), desc="Processing annotations"):
        for box in annotation["boxes"]:
            if "model_probability" in box:
                model_probability = box["model_probability"]
                if model_probability not in model_probability_distribution:
                    model_probability_distribution[model_probability] = []
                model_probability_distribution[model_probability].append(box)

    # Define bins (0-10, 10-20, ..., 90-100)
    bin_labels = [f"{i*10}-{(i+1)*10}" for i in range(10)]
    x = np.arange(len(bin_labels))  # X-axis positions for bars
    width = 0.4  # Bar width

    model_probability_distribution = dict(sorted(model_probability_distribution.items(), key=lambda x: x[0]))
    for model_probability, boxes in tqdm(model_probability_distribution.items(), desc="Processing scores"):
        gpt_scores = [0] * 10
        qwen_scores = [0] * 10

        for box in boxes:
            if "gpt_score" and "qwen_score" in box:
                gpt_class = box["gpt_class"]
                qwen_class = box["qwen_class"]

                gpt_student_dict = box["gpt_score"]
                gpt_student_top_pred = gpt_student_dict["top_preds"][0][0]
                gpt_student_top_pred_score = gpt_student_dict["top_preds"][0][1]

                qwen_student_dict = box["qwen_score"]
                qwen_student_top_pred = qwen_student_dict["top_preds"][0][0]
                qwen_student_top_pred_score = qwen_student_dict["top_preds"][0][1]

                # only keep scores if the 4 models agree
                if gpt_class == qwen_class and gpt_class == gpt_student_top_pred and gpt_class == qwen_student_top_pred:
                    gpt_score = gpt_student_top_pred_score * 100
                    bin_index = min(int(gpt_score / 10), 9)
                    gpt_scores[bin_index] += 1

                    qwen_score = qwen_student_top_pred_score * 100
                    bin_index = min(int(qwen_score / 10), 9)
                    qwen_scores[bin_index] += 1
                    
        # Side-by-side bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, gpt_scores, width=width, label="GPT", color="red", alpha=0.7)
        plt.bar(x + width/2, qwen_scores, width=width, label="Qwen", color="blue", alpha=0.7)
        
        # Label formatting
        plt.xticks(x, bin_labels, rotation=45)
        plt.xlabel("Student Score Bins")
        plt.ylabel("Frequency")
        plt.title(f"Model Probability: {model_probability}")
        plt.legend()
        plt.tight_layout()
        plt.show()
