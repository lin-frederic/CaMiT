import os
import json
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def blur_faces(task):
    image_path, faces = task
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        x1 = max(0, min(width-1, x1))
        y1 = max(0, min(height-1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            print(f"Invalid face bounding box: {face['bbox']}")
            continue
        
        # Blur the face
        face_region = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
        image[y1:y2, x1:x2] = blurred_face
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Save the blurred image
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_annotations", type=str, help="Path to pretrain annotations file", default="pretrain_annotations_with_faces.json")
    parser.add_argument("--pretrain_data", type=str, help="Path to pretrain data", default=os.path.join(os.environ['HOME'], 'pretraining_car_time'))
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--num_jobs", type=int, default=8)
    args = parser.parse_args()

    with open(args.pretrain_annotations, 'r') as f:
        annotations = json.load(f)
    
    all_tasks = []
    for time, images in annotations.items():
        for image_id, image_data in images.items():
            image_path = os.path.join(args.pretrain_data, time, image_id+".jpg")
            all_tasks.append((image_path, image_data["faces"]))
    
    total_images = len(all_tasks)
    print(f"Total number of images to process: {total_images}")

    # split into jobs
    images_per_jobs = total_images // args.num_jobs
    start = args.job_id * images_per_jobs
    end = (args.job_id + 1) * images_per_jobs if args.job_id != args.num_jobs - 1 else total_images
    job_tasks = all_tasks[start:end]
    print(f"Number of images to process for job {args.job_id}: {len(job_tasks)}")
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        list(tqdm(executor.map(blur_faces, job_tasks), total=len(job_tasks), desc="Blurring faces"))