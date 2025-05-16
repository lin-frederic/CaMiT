import os
import json
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
path = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(path, "sam2")
if sam2_path not in sys.path:
    sys.path.append(sam2_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


sam2_checkpoint = os.path.join(sam2_path, "checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def containment_ratio(box1, mask2):
    # Calculate area of mask2 / area of box1
    # This is to check if the car in the mask2 is hidding the car in the box1 (high occlusion)

    x1,y1,x2,y2 = box1
    mask1 = np.zeros_like(mask2, dtype=np.uint8)
    mask1[y1:y2,x1:x2] = 1
    mask2 = mask2.astype(np.uint8)

    # get area of mask2 in box1
    mask2_in_box1 = mask2 * mask1 # intersection between mask2 and box1
    mask2_area = np.sum(mask2_in_box1)


    box1_area = (x2-x1)*(y2-y1)

    return mask2_area/box1_area if box1_area > 0 else 0

def mask_occupancy_ratio(mask,box):
    # Compute the occupancy ratio of the mask in the box
    x1,y1,x2,y2 = box
    mask = mask.astype(np.uint8)
    mask_box = mask[y1:y2,x1:x2]
    mask_area = np.sum(mask_box)
    box_area = (x2-x1)*(y2-y1)
    return mask_area/box_area if box_area > 0 else 0

def get_color_mask(mask, color, alpha=0.5):
    """Apply a given color to a binary mask for visualization."""
    mask = mask.astype(np.uint8)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Apply color where mask is 1
    for i in range(3):
        color_mask[:, :, i] = mask * color[i]
    
    return color_mask, mask * alpha

def draw_mask_contour(image,mask,color,thickness=2):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image

def visualize_masks(image, box_i, mask_i, box_j, mask_j, ratio, alpha=0.5):
    """Show cropped region of box_i with masks overlaid."""
    x1, y1, x2, y2 = box_i  # Crop only box_i region
    cropped_image = image[y1:y2, x1:x2].copy()
    mask_i = mask_i[y1:y2, x1:x2]
    mask_j = mask_j[y1:y2, x1:x2]

    cropped_image = draw_mask_contour(cropped_image, mask_i, (0, 255, 0)) # Green for box_i
    cropped_image = draw_mask_contour(cropped_image, mask_j, (255, 0, 0)) # Red for mask_j
    return cropped_image

    """# Get masks for box_i and intersection of mask_j with box_i
    color_mask1, alpha1 = get_color_mask(mask_i[y1:y2, x1:x2], (0, 255, 0), alpha) # Green for box_i
    color_mask2, alpha2 = get_color_mask(mask_j[y1:y2, x1:x2], (255, 0, 0), alpha) # Red for mask_j

    # Convert cropped image to float32 for blending
    cropped_image = cropped_image.astype(np.float32) / 255.0
    color_mask1 = color_mask1.astype(np.float32) / 255.0
    color_mask2 = color_mask2.astype(np.float32) / 255.0
    
    # Alpha blending: foreground * alpha + background * (1 - alpha)
    overlay = cropped_image * (1-alpha1[:,:,None]) + color_mask1 * alpha1[:,:,None]
    overlay = overlay * (1-alpha2[:,:,None]) + color_mask2 * alpha2[:,:,None]

    # convert back to uint8
    overlay = (overlay * 255).astype(np.uint8)
 
    return overlay"""

def intersect(box1, box2):
    cx1,cy1,w1,h1 = box1
    cx2,cy2,w2,h2 = box2
    x1_min, x1_max = cx1-w1/2, cx1+w1/2
    y1_min, y1_max = cy1-h1/2, cy1+h1/2

    x2_min, x2_max = cx2-w2/2, cx2+w2/2
    y2_min, y2_max = cy2-h2/2, cy2+h2/2

    if x1_min > x2_max or x1_max < x2_min:
        return False
    if y1_min > y2_max or y1_max < y2_min:
        return False
    return True
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to the train annotations", default="outputs/train_gpt_annotations_with_unknown.json") 
    parser.add_argument("--test_annotations", type=str, help="Path to the test annotations", default="outputs/test_gpt_annotations_with_unknown.json")
    args = parser.parse_args()

    with open(args.train_annotations) as f:
        train_annotations = json.load(f)
    
    with open(args.test_annotations) as f:
        test_annotations = json.load(f)

    grouped_annotations = {**test_annotations, **train_annotations} # test then train
    count = 0
    images_with_duplicates = {}

    for image_id, image_data in tqdm(grouped_annotations.items()):
        boxes = image_data["boxes"]
        boxes = [box for box in boxes if "gpt_class" in box]
        # process only images with more than one box of the same class
        if len(boxes) <= 1:
            continue
        has_duplicates = False
        for i, box in enumerate(boxes):
            for j in range(i+1, len(boxes)):
                if box["gpt_class"] == boxes[j]["gpt_class"] and intersect(box["box"], boxes[j]["box"]):
                    has_duplicates = True
                    break
        
        if has_duplicates:
            images_with_duplicates[image_id] = image_data
        count += 1
    
    print(f"Images with more than one box: {count}")
    print(f"Images with duplicates: {len(images_with_duplicates)}")
    exit()

    os.makedirs("clean_outlier/occlusion_analysis", exist_ok=True)
    for i in range(6):
        os.makedirs(f"clean_outlier/occlusion_analysis/bin_{i}", exist_ok=True)
    processed_images = set()
    bins_count = [0]*6
    for i in range(6):
        bin_images = os.listdir(f"clean_outlier/occlusion_analysis/bin_{i}")
        bins_count[i] = len(bin_images)
        for image in bin_images:
            processed_images.add(image.split("_")[0]) # image: imageid_ratio.jpg

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    for image_id, image_data in tqdm(images_with_duplicates.items()):
        if image_id in processed_images:
            continue
        
        image_path = image_data["image_path"].replace("/home/users/flin","/home/fredericlin")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = image_data["boxes"]
        boxes = [box for box in boxes if "gpt_class" in box]
        # only keep a box if it shares the same class with another box and they intersect
        new_boxes = []
        has_seen = [False]*len(boxes) 
        for i, box in enumerate(boxes):
            for j in range(i+1, len(boxes)):
                if has_seen[i] or has_seen[j]:
                    continue
                """if box["gpt_class"]!=boxes[j]["gpt_class"] and intersect(box["box"], boxes[j]["box"]):
                    # check intersecting box with different class
                    cx1,cy1,w1,h1 = box["box"]
                    cx2,cy2,w2,h2 = boxes[j]["box"]
                    x1_min, x1_max = cx1-w1/2, cx1+w1/2
                    y1_min, y1_max = cy1-h1/2, cy1+h1/2
                    x2_min, x2_max = cx2-w2/2, cx2+w2/2
                    y2_min, y2_max = cy2-h2/2, cy2+h2/2

                    cv2.rectangle(image, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (0,255,0), 2)
                    cv2.rectangle(image, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255,0,0), 2)

                    plt.imshow(image)
                    plt.show()
                continue"""

                if box["gpt_class"] == boxes[j]["gpt_class"] and intersect(box["box"], boxes[j]["box"]):
                    new_boxes.append(box)
                    new_boxes.append(boxes[j])
                    has_seen[i] = True
                    has_seen[j] = True
                    break
        #continue
        boxes = new_boxes
        predictor.set_image(image)

        # convert boxes to format [x1,y1,x2,y2]
        input_boxes = []
        for box in boxes:
            cx,cy,w,h = box["box"]
            input_boxes.append([int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)])
        input_boxes = np.array(input_boxes)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=True
        )
        for i, box_i in enumerate(input_boxes): # check if there is another car that is occluding the car in box_i
            mask_i = masks[i][np.argmax(scores[i])]
            for j, box_j in enumerate(input_boxes):
                if i==j:
                    continue
                mask_j = masks[j][np.argmax(scores[j])]
                ratio = containment_ratio(box_i, mask_j)
        
                # add to the corresponding bin if the bin is not full (ie < 100 examples)
                if ratio >= 0.5:
                    bin_index = 5
                else:
                    bin_index = int(ratio*10)

                if bins_count[bin_index] < 100:
                    bins_count[bin_index] += 1
                    overlay = visualize_masks(image, box_i, mask_i, box_j, mask_j, ratio, alpha=0.4)
                    cv2.imwrite(f"clean_outlier/occlusion_analysis/bin_{bin_index}/{image_id}_{ratio:.2f}.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                if all([count >= 100 for count in bins_count]):
                    print(f"Found enough examples for each bin")
                    exit()
                
