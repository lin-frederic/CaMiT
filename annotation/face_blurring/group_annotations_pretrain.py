import os
import argparse

import json
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default='face_blurring/insightface_results_pretrain', help="path to the results directory")
    parser.add_argument("--results_supplementary", type=str, default='face_blurring/insightface_results_pretrain_supplementary', help="path to the supplementary results directory") 
    parser.add_argument("--results_missing", type=str, default='face_blurring/insightface_results_pretrain_missing', help="path to the missing results directory")

    parser.add_argument("--pretrain_annotations", type=str, default='pretrain_annotations.json', help="path to the pretrain annotations file")
    args = parser.parse_args()

    face_annotations = {}

    for annotation_file in tqdm(os.listdir(args.results)):
        if annotation_file.endswith('.json'):
            with open(os.path.join(args.results, annotation_file), 'r') as f:
                annotations = json.load(f)
            for image_name, image_annotations in annotations.items():
                face_annotations[image_name] = image_annotations

    if os.path.exists(args.results_supplementary):
        for annotation_file in tqdm(os.listdir(args.results_supplementary)):
            if annotation_file.endswith('.json'):
                with open(os.path.join(args.results_supplementary, annotation_file), 'r') as f:
                    annotations = json.load(f)
                for image_name, image_annotations in annotations.items():
                    face_annotations[image_name] = image_annotations
    
    if os.path.exists(args.results_missing):
        for annotation_file in tqdm(os.listdir(args.results_missing)):
            if annotation_file.endswith('.json'):
                with open(os.path.join(args.results_missing, annotation_file), 'r') as f:
                    annotations = json.load(f)
                for image_name, image_annotations in annotations.items():
                    face_annotations[image_name] = image_annotations

    print(f"Total number of images with face annotations: {len(face_annotations)}")
    # Load the pretrain annotations
    with open(args.pretrain_annotations, 'r') as f:
        pretrain_annotations = json.load(f)
    
    new_pretrain_annotations = {}
    for time in tqdm(pretrain_annotations.keys()):
        if time not in new_pretrain_annotations:
            new_pretrain_annotations[time] = {}
        for image_id, image_data in pretrain_annotations[time].items():
            image_data['faces'] = face_annotations[image_id]
            new_pretrain_annotations[time][image_id] = image_data
    # Save the new pretrain annotations
    with open('pretrain_annotations_with_faces.json', 'w') as f:
        json.dump(new_pretrain_annotations, f, indent=4)

  