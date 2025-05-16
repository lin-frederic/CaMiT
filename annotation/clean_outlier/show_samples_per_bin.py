import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_annotations", type=str, help="Path to the student annotations", default="outputs/student_annotations_clean.json")
    args = parser.parse_args()



    with open(args.student_annotations) as f:
        student_annotations = json.load(f)
    
    

    for image_id, annotation in student_annotations.items():
        for box in annotation["boxes"]:
            if "model_probability" in box and "gpt_score" in box and "qwen_score" in box:
                model_probability = box["model_probability"]

                if model_probability < 85:
                    continue

                gpt_class = box["gpt_class"]
                qwen_class = box["qwen_class"]
                gpt_student_dict = box["gpt_score"]
                gpt_student_top_pred = gpt_student_dict["top_preds"][0][0]
                gpt_student_top_pred_score = gpt_student_dict["top_preds"][0][1]

                qwen_student_dict = box["qwen_score"]
                qwen_student_top_pred = qwen_student_dict["top_preds"][0][0]
                qwen_student_top_pred_score = qwen_student_dict["top_preds"][0][1]

                if gpt_class == qwen_class and gpt_class == gpt_student_top_pred and gpt_class == qwen_student_top_pred:
                    gpt_score = gpt_student_top_pred_score * 100
                    qwen_score = qwen_student_top_pred_score * 100
                    # check samples per bin
                    if gpt_score < 80 or qwen_score < 80:
                        continue

                    image_path = annotation["image_path"].replace("/home/users/flin","/home/fredericlin")
                    image = cv2.imread(image_path)

                    cx, cy, w, h = box["box"]
                    x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    crop = image[y1:y2, x1:x2]
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    plt.title(f"M: {gpt_class} ({model_probability}%) GPT: {gpt_score:.2f} QWEN: {qwen_score:.2f}")
                    plt.show()

