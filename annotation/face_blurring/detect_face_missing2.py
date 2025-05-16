import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_USE_THREAD_AFFINITY'] = '0'


def pad_to_square(image):
    h,w,_ = image.shape
    size = max(h,w)
    top = (size-h)//2
    bottom = size-h-top
    left = (size-w)//2
    right = size-w-left
    image_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return image_padded, top, left, size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default='pretrain_annotations_supplementary_final_2008_2023.json', help="path to the pretrain annotations file")
    parser.add_argument("--missing_images", type=str, default='face_blurring/missing_images2.txt', help="path to the missing images file")
    parser.add_argument("--save_dir", type=str, default='face_blurring/insightface_results_pretrain_supplementary_final_2008_2023_missing', help="path to the output directory")
    parser.add_argument('--job', type=int, default=0, help='job id')
    parser.add_argument("--num_job", type=int, default=8, help="number of jobs to split the dataset into")
    args = parser.parse_args()


    with open(args.annotations, 'r') as f:
        annotations = json.load(f)

    all_images = []
    for time in annotations.keys():
        for image_id, image_data in annotations[time].items():
            all_images.append((time, image_id))
    
    with open(args.missing_images, 'r') as f:
        missing_images = f.readlines()
    

    missing_images = [x.strip() for x in missing_images]
    missing_images = set(missing_images)
    all_images = [(time, image_id) for time, image_id in all_images if image_id in missing_images]
    print(f"Total number of images with missing face annotations: {len(all_images)}")

    # split the dataset into n jobs
    num_images = len(all_images)
    images_per_job = num_images // args.num_job
    start_index = args.job * images_per_job
    end_index = (args.job + 1) * images_per_job if args.job != args.num_job - 1 else num_images
    all_images = all_images[start_index:end_index]
    job_annotations = {}
    for time, image_id in all_images:
        job_annotations[image_id] = annotations[time][image_id]

    print(f"Processing {len(all_images)} images for job {args.job}...")

    # Load the model
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    face_annotations = {}
    for image_name, image_info in tqdm(job_annotations.items()):
        #image_path = image_info['image_path'].replace("/home/users/flin", "/home/fredericlin")
        image_path = image_info["image_path"]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        padded_img, top, left, size = pad_to_square(img)
        resized_img = cv2.resize(padded_img, (640, 640))
        scale = size / 640
        faces = app.get(resized_img)
        # retrieve results
        retrieved_faces = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox # coordinates in the resized image
            x1 = int(x1*scale - left)
            y1 = int(y1*scale - top)
            x2 = int(x2*scale - left)
            y2 = int(y2*scale - top)
            bbox = [x1, y1, x2, y2]
            det_score = face.det_score
            retrieved_faces.append({
                'bbox': bbox,
                'det_score': float(det_score),
            })

        face_annotations[image_name] = retrieved_faces
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f'face_annotations_{args.job}.json'), 'w') as f:
        json.dump(face_annotations, f, indent=4)
