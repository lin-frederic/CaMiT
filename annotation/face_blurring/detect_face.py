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
    parser.add_argument('--output', type=str, default='insightface_results/final_test_annotations.json', help='path to the output file')
    parser.add_argument('--job', type=int, default=0, help='job id')
    parser.add_argument("--num_job", type=int, default=8, help="number of jobs to split the dataset into")
    parser.add_argument("--split", choices=['train', 'test'], default='test', help="split to process")
    args = parser.parse_args()


    if args.split == 'train':
        args.annotation = '../final_train_annotations.json'
        args.output = 'insightface_results/final_train_annotations_{}.json'.format(args.job)
    elif args.split == 'test':
        args.annotation = '../final_test_annotations.json'
        args.output = 'insightface_results/final_test_annotations_{}.json'.format(args.job)
    else:
        raise ValueError("split must be either 'train' or 'test'")
    with open(args.annotation, 'r') as f:
        annotations = json.load(f)
    
    # split the dataset into n jobs
    image_names = list(annotations.keys())
    num_images = len(image_names)
    images_per_job = num_images // args.num_job
    start_index = args.job * images_per_job
    end_index = (args.job + 1) * images_per_job if args.job != args.num_job - 1 else num_images

    image_names = image_names[start_index:end_index]
    annotations = {k: annotations[k] for k in image_names}
    print(f"Processing {len(annotations)} images for job {args.job}...")

    # Load the model
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)


    for image_name, image_info in tqdm(annotations.items()):
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
        
        annotations[image_name]['faces'] = retrieved_faces


    os.makedirs('insightface_results', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(annotations, f, indent=4)
