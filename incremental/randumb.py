import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import SupervisedTimeDataset, ClassOrder
from models import get_model, load_embeddings, get_model_dim
from utils import parse_args, get_years, get_collate_fn, target2onehot
from sklearn.kernel_approximation import RBFSampler

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
    return prototypes

def compute_global_prototype(embeddings):
    """
    Compute the global prototype
    embeddings: (N, D)
    """
    global_prototype = torch.mean(embeddings, dim=0) # (D,)
    return global_prototype

def compute_second_moment(embeddings):
    """
    Compute the second moment of the embeddings
    embeddings: (N, D)
    """
    second_moment = embeddings.T @ embeddings / embeddings.shape[0]  # (D, D)
    return second_moment


def update_statistics(old_prototype, old_count, new_prototype, new_count):
    """
    Update the statistics of the model
    prototypes: {label: (D,)}
    """
    n_old = old_count
    n_new = new_count
    n_total = n_old + n_new
    new_prototype = (n_old * old_prototype + n_new * new_prototype) / n_total
    return new_prototype, n_total

def update_global_statistics(old_global_prototype, old_global_second_moment, old_count, new_global_prototype, new_global_second_moment, new_count):
    """
    Update the global prototype and second moment incrementally.
    """
    n_old = old_count
    n_new = new_count
    n_total = n_old + n_new

    updated_global_prototype = (n_old * old_global_prototype + n_new * new_global_prototype) / n_total
    updated_global_second_moment = (n_old * old_global_second_moment + n_new * new_global_second_moment) / n_total

    return updated_global_prototype, updated_global_second_moment, n_total

def compute_covariance(mean, second_moment, count):
    """
    Compute the covariance matrix using the second moment and mean.
    Cov(X) = E[X^2] - E[X]^2
    """
    covariance = second_moment - torch.outer(mean, mean)
    # OAS shrinkage (from sklearn - can't do oas directly from covariance)
    mu = torch.trace(covariance) / covariance.shape[0]
    alpha = torch.mean(covariance**2)
    num = alpha + mu**2
    den = (count + 1) * (alpha - mu**2/covariance.shape[0])
    shrinkage = 1. if den == 0 else min(1., num / den)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * mu * torch.eye(covariance.shape[0])
    return covariance

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

def fecam(prototypes, inv_covariance, test_embeddings, test_labels, device="cuda", batch_size=1024):
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
    #test_embeddings = F.normalize(test_embeddings, dim=1)
    test_labels = test_labels
    preds = []
    inv_cov = inv_covariance.to(device)  # (D, D)

    with torch.no_grad():
        for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Batch processing"):
            batch_embeddings = test_embeddings[i:i + batch_size].to(device)  # (B, D)
            distances = []
            for label in proto_labels:
                prototype = prototypes[label].to(device)  # (D,)
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
    assert args.num_jobs == 1, "Only one job is supported for RanPAC"
    model, transform, model_name, device = get_model(args)
    train_years, test_years = get_years(args)  
    results = {}
    os.makedirs("features", exist_ok=True)
    os.makedirs(f"features/{model_name}", exist_ok=True)

    averaged_prototypes = {}
    averaged_global_prototype = None
    averaged_global_second_moment = None
    global_count = 0
    counts = {}

    embed_dim = 10000
    rbf_sampler = None

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

            if rbf_sampler is None:
                rbf_sampler = RBFSampler(gamma='scale', n_components=embed_dim)
                rbf_sampler.fit(train_embeddings.cpu().numpy())
            train_embeddings = rbf_sampler.transform(train_embeddings.cpu().numpy())
            train_embeddings = torch.tensor(train_embeddings)
            prototypes = compute_prototypes(train_embeddings, train_labels)

            # Compute the global prototype and second moment
            global_prototype = compute_global_prototype(train_embeddings)
            global_second_moment = compute_second_moment(train_embeddings)

            del train_embeddings

            # Update the global statistics
            if averaged_global_prototype is None:
                averaged_global_prototype = global_prototype
                averaged_global_second_moment = global_second_moment
                global_count = len(train_labels)
            else:
                averaged_global_prototype, averaged_global_second_moment, global_count = update_global_statistics(
                    averaged_global_prototype, averaged_global_second_moment, global_count,
                    global_prototype, global_second_moment, len(train_labels)
                )
            
            del global_prototype, global_second_moment

            inv_covariance = compute_covariance(averaged_global_prototype, averaged_global_second_moment, global_count)
            inv_covariance = torch.linalg.inv(inv_covariance)

             # Update the class prototypes and counts incrementally
            for label in prototypes.keys():
                if label not in averaged_prototypes:
                    averaged_prototypes[label] = prototypes[label]
                    counts[label] = len([l for l in train_labels if l == label])
                else:
                    averaged_prototypes[label], counts[label] = update_statistics(averaged_prototypes[label], 
                                                                                  counts[label], 
                                                                                  prototypes[label],
                                                                                  len([l for l in train_labels if l == label]))
            

            del train_labels, prototypes
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
                test_embeddings = rbf_sampler.transform(test_embeddings.cpu().numpy())
                test_embeddings = torch.tensor(test_embeddings)
                # Compute the accuracy
                acc = fecam(averaged_prototypes, inv_covariance, test_embeddings, test_labels, device=device)
                results[train_year][test_year] = float(acc)
                print(f"Train year: {train_year}, Test year: {test_year}, NCM accuracy: {acc:.4f}")
                pbar.update(1)
    # Save results
    os.makedirs("randumb_results", exist_ok=True)
    os.makedirs(f"randumb_results/{model_name}", exist_ok=True)
    with open(f"randumb_results/{model_name}/randumb_results_{args.job_id}.json", "w") as f:
        json.dump(results, f, indent=4)