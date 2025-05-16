import os
from tqdm import tqdm
import json
import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SupervisedTimeDataset

from models import get_model, load_embeddings
from utils import parse_args, get_years, get_collate_fn
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Optional

def compute_prototypes_and_covariances(embeddings, labels):
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
        covariances[label] = cov_matrix
    return prototypes, covariances

def compute_fid(prototypeA, prototypeB, covarianceA, covarianceB):
    """
    Compute the FID between two sets of prototypes and covariances
    """
    prototypeA = prototypeA.cpu().numpy()
    prototypeB = prototypeB.cpu().numpy()
    covarianceA = covarianceA.cpu().numpy()
    covarianceB = covarianceB.cpu().numpy()

    diff = prototypeA - prototypeB
    covmean, _ = sqrtm(covarianceA @ covarianceB, disp=False)
    if not np.isfinite(covmean).all():
        msg = "FID calculation produces singular product; adding %s to diagonal of cov estimates" % 1e-6
        print(msg)
        offset = np.eye(covarianceA.shape[0]) * 1e-6
        covmean = sqrtm((covarianceA + offset) @ (covarianceB + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component of FID is not zero.")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(covarianceA) + np.trace(covarianceB) - 2 * tr_covmean

    return fid


def polynomial_mmd_fx_fy(
    fx: torch.Tensor,
    fy: torch.Tensor,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0
) -> torch.Tensor:
    m, n = fx.shape[0], fy.shape[0]
    D    = fx.shape[1]
    if gamma is None:
        gamma = 1.0 / D

    Kxx = (gamma * fx @ fx.T + coef0).pow(degree)
    Kyy = (gamma * fy @ fy.T + coef0).pow(degree)
    Kxy = (gamma * fx @ fy.T + coef0).pow(degree)

    sum_xx = Kxx.sum() - Kxx.diagonal().sum()
    sum_yy = Kyy.sum() - Kyy.diagonal().sum()

    mmd2 = ( sum_xx / (m * (m - 1))
           + sum_yy / (n * (n - 1))
           - 2.0 * Kxy.mean() )
    return mmd2

if __name__=="__main__":
    args = parse_args()
    assert args.num_jobs == 1, "Only one job is supported for embedding shift"
    model, transform, model_name, device = get_model(args)
    train_years, test_years = get_years(args)  
    os.makedirs("features", exist_ok=True)
    os.makedirs(f"features/{model_name}", exist_ok=True)
    all_embeddings = {}
    for train_year in tqdm(train_years, desc="Processing train years"):
        features_path = os.path.join(f"features/{model_name}", f"features_train_{train_year}.pt")
        if not os.path.exists(features_path):
            train_dataset = SupervisedTimeDataset(args.train_annotations, args.train_images_dir, train_year, transform=transform)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=get_collate_fn(args.model))
        else:
            train_dataset = None
            train_dataloader = None
        train_embeddings, train_labels = load_embeddings(model, train_dataloader, device, features_path, model_name)

        test_features_path = os.path.join(f"features/{model_name}", f"features_test_{train_year}.pt")
        if not os.path.exists(test_features_path):
            test_dataset = SupervisedTimeDataset(args.test_annotations, args.test_images_dir, train_year, transform=transform)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=get_collate_fn(args.model))
        else:
            test_dataset = None
            test_dataloader = None
        test_embeddings, test_labels = load_embeddings(model, test_dataloader, device, test_features_path, model_name)

        embeddings = torch.cat([train_embeddings, test_embeddings], dim=0)
        # normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        labels = train_labels + test_labels

        # Store embeddings by label
        for i, label in enumerate(labels):
            if label not in all_embeddings:
                all_embeddings[label] = {}
            if train_year not in all_embeddings[label]:
                all_embeddings[label][train_year] = []
            all_embeddings[label][train_year].append(embeddings[i])
        
    
    os.makedirs("embedding_shift_results", exist_ok=True)
    os.makedirs(f"embedding_shift_results/{model_name}", exist_ok=True)

    for label in tqdm(all_embeddings.keys(), desc="Computing KID"):
        time_shift_label = {}
        with tqdm(total=len(all_embeddings[label])**2) as pbar:
            for i, train_year in enumerate(all_embeddings[label].keys()):
                embeddingsA = torch.stack(all_embeddings[label][train_year])
                for j, train_yearB in enumerate(all_embeddings[label].keys()):
                    if i > j:
                        pbar.update(1)
                        continue
                    embeddingsB = torch.stack(all_embeddings[label][train_yearB])
                    
                    # Compute KID
                    kid = polynomial_mmd_fx_fy(embeddingsA, embeddingsB)
                    if train_year not in time_shift_label:
                        time_shift_label[train_year] = {}
                    if train_yearB not in time_shift_label:
                        time_shift_label[train_yearB] = {}
                    time_shift_label[train_year][train_yearB] = float(kid)
                    time_shift_label[train_yearB][train_year] = float(kid)
                    pbar.update(1)
        # save results for each label
        with open(os.path.join(f"embedding_shift_results/{model_name}", f"results_{label}.json"), "w") as f:
            json.dump(time_shift_label, f, indent=4)
        
    
                    
    
