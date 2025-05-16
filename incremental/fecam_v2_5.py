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

def shrink_cov(cov,alpha1=1,alpha2=0):
    """
    Shrink the covariance matrix to make it more stable.
    cov: (D, D)
    """
    # Compute the mean of the diagonal
    diag_mean = torch.mean(torch.diag(cov))
    off_diag = cov - torch.diag(torch.diag(cov))
    # Compute the mean of the off-diagonal elements
    off_diag_mean = torch.mean(off_diag[off_diag != 0])
    identity = torch.eye(cov.shape[0])
    cov = cov + (alpha1 * diag_mean* identity) + (alpha2 * off_diag_mean * (1 - identity))
    return cov

def normalize_cov(cov):
    """
    Normalize the covariance matrix : Sigma[i,j] = Sigma[i,j]/(sigma[i]*sigma[j])
    where sigma[i] = sqrt(Sigma[i,i])
    cov: (D, D)
    """
    diag = torch.sqrt(torch.diag(cov))
    diag_inv = 1 / diag
    cov = cov * diag_inv[:, None] * diag_inv[None, :]
    return cov


def compute_prototypes_and_covariances(embeddings, labels,alpha1=1,alpha2=0):
    """
    Compute the prototypes for each class
    embeddings: (N, D)
    labels: (N,)
    """
    unique_labels = set(labels)
    prototypes = {}
    covariances = {}
    for label in tqdm(unique_labels, desc="Computing prototypes"):
        indices = [i for i, l in enumerate(labels) if l == label]
        embeddings[indices] = F.normalize(embeddings[indices], dim=1)
        # Compute covariance matrix
        prototypes[label] = torch.mean(embeddings[indices], dim=0) # (D,)
        cov_matrix = torch.cov(embeddings[indices].T)
        cov_matrix = shrink_cov(cov_matrix,alpha1=alpha1,alpha2=alpha2)
        #cov_matrix = normalize_cov(cov_matrix)
        covariances[label] = cov_matrix
    return prototypes, covariances

def update_statistics(old_prototype, old_count, new_prototype, new_count):
    """
    Update the prototype and count incrementally.
    """
    total_count = old_count + new_count
    updated_prototype = (old_count * old_prototype + new_count * new_prototype) / total_count
    return updated_prototype, total_count

def batched_mahalanobis(batch_embeddings, prototype, inv_cov):
    """
    Compute the Mahalanobis distance between each test embedding and the prototype.
    Args:
        batch_embeddings: (B, D) tensor
        prototype: (D,) tensor
        inv_cov: (D, D) tensor
    """
    # Compute the difference between the test embedding and the prototype
    diff = batch_embeddings - prototype  # (B, D)
    # Compute the Mahalanobis distance
    left = torch.matmul(diff, inv_cov)  # (B, D)
    dists = torch.matmul(left, diff.T)  # (B, B)
    return dists.diagonal()  # (B,)

def fecam(prototypes, inv_covariances, test_embeddings, test_labels, device="cuda", batch_size=1024):
    """
    Compute FeCAM accuracy using Mahalanobis distance.
    Args:
        prototypes: dict {label: (D,)}
        covariances: dict {label: (D, D)}
        test_embeddings: (N, D) tensor
        test_labels: (N,) tensor
    Returns:    
        accuracy: float
    """
    # Prepare prototype tensor
    proto_labels = list(prototypes.keys()) # (C,)

    # Normalize test_embeddings
    test_embeddings = F.normalize(test_embeddings, dim=1)
    test_labels = test_labels
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Batch processing"):
            batch_embeddings = test_embeddings[i:i + batch_size].to(device)  # (B, D)
            distances = []
            for label in inv_covariances.keys():
                prototype = prototypes[label].to(device)  # (D,)
                inv_cov = inv_covariances[label].to(device)  # (D, D)
                dists = batched_mahalanobis(batch_embeddings, prototype, inv_cov)  # (B,)
                distances.append(dists)
            distances = torch.stack(distances, dim=1)  # (B, C)
            preds_batch = torch.argmin(distances, dim=1)  # (B,)
            preds.append(preds_batch)
        preds = torch.cat(preds, dim=0)  # (N,)

    # Convert predictions to labels
    pred_labels = [proto_labels[i] for i in preds]
    correct = sum([int(p == t) for p, t in zip(pred_labels, test_labels)])
    acc = correct / len(test_labels)

    return acc

if __name__=="__main__":
    args = parse_args()
    assert args.num_jobs == 1, "Only one job is supported for FeCAMv2"
    model, transform, model_name, device = get_model(args)

    results = {}
    train_years, test_years = get_years(args)

    os.makedirs("features", exist_ok=True)
    os.makedirs(f"features/{model_name}", exist_ok=True)
    
    averaged_prototypes = {}
    averaged_covariances = {}
    counts = {}

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
            prototypes, covariances = compute_prototypes_and_covariances(train_embeddings, train_labels, args.alpha1, args.alpha2)

            # compute inverse of covariance matrix for each class
            inv_covariances = {}
            for label in prototypes.keys():
                if label not in averaged_covariances:
                    averaged_prototypes[label] = prototypes[label]
                    averaged_covariances[label] = covariances[label]
                    counts[label] = len([l for l in train_labels if l == label])
                    
                else:
                    averaged_covariances[label] = (averaged_covariances[label] + covariances[label]) / 2
                    averaged_prototypes[label], counts[label] = update_statistics(averaged_prototypes[label], counts[label], prototypes[label], len([l for l in train_labels if l == label]))
                # average covariance matrices might not be positive definite so regularize
                #inv_covariances[label] = averaged_covariances[label] + torch.eye(averaged_covariances[label].shape[0]) * 1e-6
                inv_covariances[label] = normalize_cov(averaged_covariances[label])
                inv_covariances[label] = torch.linalg.pinv(averaged_covariances[label])
                #inv_covariances[label] = torch.linalg.pinv(covariances[label])
                #covariances[label] = torch.inverse(covariances[label])

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
                acc = fecam(averaged_prototypes, inv_covariances, test_embeddings, test_labels)
                #acc = fecam(prototypes, inv_covariances, test_embeddings, test_labels)
                results[train_year][test_year] = acc
                print(f"Train year: {train_year}, Test year: {test_year}, Accuracy: {acc:.4f}")
                pbar.update(1)
                
    # Save results
    os.makedirs("fecamv2_5_results", exist_ok=True)
    os.makedirs(f"fecamv2_5_results/{model_name}", exist_ok=True)
    with open(os.path.join(f"fecamv2_5_results/{model_name}", f"results_{args.job_id}.json"), "w") as f:
        json.dump(results, f, indent=4)