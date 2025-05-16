import os
import json
import argparse

from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default='face_blurring/insightface_results_pretrain_supplementary_final_2008_2023', help="path to the results directory")
    parser.add_argument("--pretrain_annotations", type=str, default='pretrain_annotations_supplementary_final_2008_2023.json', help="path to the pretrain annotations file")
    args = parser.parse_args()

    face_annotations = {}

    for annotation_file in tqdm(os.listdir(args.results)):
        if annotation_file.endswith('.json'):
            with open(os.path.join(args.results, annotation_file), 'r') as f:
                annotations = json.load(f)
            for image_name, image_annotations in annotations.items():
                face_annotations[image_name] = image_annotations
    print(f"Total number of images with face annotations: {len(face_annotations)}") 

    # Load the pretrain annotations
    with open(args.pretrain_annotations, 'r') as f:
        pretrain_annotations = json.load(f)
    
    pretrain_images = set()
    for time in tqdm(pretrain_annotations.keys()):
        
        for image_id, image_data in pretrain_annotations[time].items():
            pretrain_images.add(image_id)
    print(f"Total number of images in pretrain annotations: {len(pretrain_images)}")    

    missing_images = []
    for image_id in tqdm(pretrain_images):
        if image_id not in face_annotations:
            missing_images.append(image_id)
    
    print(f"Total number of images with missing face annotations: {len(missing_images)}")
    with open('face_blurring/missing_images2.txt', 'w') as f:
        for image_id in missing_images:
            f.write(image_id + '\n')
