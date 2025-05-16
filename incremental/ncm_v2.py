import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import SupervisedTimeDataset
from models import get_model, load_embeddings
from utils import parse_args, get_years, get_collate_fn

def compute_prototypes(embeddings, labels):
    """
    Compute the prototypes for each class
    embeddings: (N, D)
    labels: (N,)
    """
    unique_labels = set(labels)
    prototypes = {}
    for label in tqdm(unique_labels, desc="Computing prototypes"):
        indices = [i for i, l in enumerate(labels) if l == label]
        # normalize the embeddings (N, D)
        #embeddings[indices] = F.normalize(embeddings[indices], dim=1)
        prototypes[label] = torch.mean(embeddings[indices], dim=0) # (D,)
    return prototypes # {label: (D,)}

def update_statistics(old_prototype, old_count, new_prototype, new_count):
    """
    Update the prototype and count incrementally.
    """
    # Update the prototype
    new_prototype = (old_count * old_prototype + new_count * new_prototype) / (old_count + new_count)
    # Update the count
    new_count = old_count + new_count
    return new_prototype, new_count

def ncm(prototypes, test_embeddings, test_labels, normalize=True):
    """
    Compute the NCM accuracy.
    Args:
        prototypes: dict {label: (D,)}
        test_embeddings: (N, D) tensor
        test_labels: (N,) tensor
        normalize: whether to use cosine similarity (normalize vectors)

    Returns:
        accuracy: float
    """
    # Prepare prototype tensor
    proto_labels = list(prototypes.keys()) # (C,)
    proto_matrix = torch.stack([prototypes[label] for label in proto_labels])  # (C, D)

    if normalize:
        proto_matrix = F.normalize(proto_matrix, dim=1)
        test_embeddings = F.normalize(test_embeddings, dim=1)

    # Compute distance matrix: (N, C)
    #dists = torch.cdist(test_embeddings, proto_matrix, p=2)  # Euclidean
    # If using cosine distance:
    dists = 1 - torch.matmul(test_embeddings, proto_matrix.T)

    preds = torch.argmin(dists, dim=1)  # (N,)
    pred_labels = [proto_labels[i] for i in preds]

    #test_labels = test_labels.tolist()
    correct = sum([int(p == t) for p, t in zip(pred_labels, test_labels)])
    acc = correct / len(test_labels)
    
    return acc




if __name__=="__main__":
    args = parse_args()
    assert args.num_jobs == 1, "Only one job is supported for NCMv2"
    model, transform, model_name, device = get_model(args)
    train_years, test_years = get_years(args)  
    results = {}
    os.makedirs("features", exist_ok=True)
    os.makedirs(f"features/{model_name}", exist_ok=True)
    averaged_prototypes = {} # E[X]
    counts = {} # n

    with tqdm(total=len(train_years) * len(test_years)) as pbar:
        for train_year in train_years:
            print(f"Processing train year {train_year}")
            features_path = os.path.join(f"features/{model_name}", f"features_train_{train_year}.pt")
            if not os.path.exists(features_path):
                train_dataset = SupervisedTimeDataset(args.train_annotations, args.train_images_dir, train_year, transform=transform)
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=get_collate_fn(args.model))
            else:
                train_dataset = None
                train_dataloader = None
            train_embeddings, train_labels = load_embeddings(model, train_dataloader, device, features_path, model_name)

            prototypes = compute_prototypes(train_embeddings, train_labels)

            # Update the averaged_prototypes and counts
            for label in prototypes.keys():
                if label not in averaged_prototypes:
                    averaged_prototypes[label] = prototypes[label]
                    counts[label] = len([l for l in train_labels if l == label])
                else:
                    averaged_prototypes[label], counts[label] = update_statistics(averaged_prototypes[label],
                                                                                  counts[label],
                                                                                  prototypes[label],
                                                                                  len([l for l in train_labels if l == label]))
            results[train_year] = {}

            for test_year in test_years:
                features_path = os.path.join(f"features/{model_name}", f"features_test_{test_year}.pt")
                if not os.path.exists(features_path):
                    test_dataset = SupervisedTimeDataset(args.test_annotations, args.test_images_dir, test_year, transform=transform)
                    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=get_collate_fn(args.model))
                else:
                    test_dataset = None
                    test_dataloader = None
                test_embeddings, test_labels = load_embeddings(model, test_dataloader, device, features_path, model_name)

                acc = ncm(averaged_prototypes, test_embeddings, test_labels)
                results[train_year][test_year] = float(acc)
                print(f"Train year: {train_year}, Test year: {test_year}, NCM accuracy: {acc:.4f}")
                pbar.update(1)
    # Save results
    os.makedirs("ncmv2_results", exist_ok=True)
    os.makedirs(f"ncmv2_results/{model_name}", exist_ok=True)
    with open(f"ncmv2_results/{model_name}/ncm_results_{args.job_id}.json", "w") as f:
        json.dump(results, f, indent=4)
