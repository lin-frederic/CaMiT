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


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-std, std)
        self.sigma.data.fill_(1.0)
    
    def forward(self, x):
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        out = out * self.sigma
        return out
    

class RanPAC(nn.Module):
    def __init__(self, model, M, num_classes):
        super(RanPAC, self).__init__()
        dim = get_model_dim(model)
        self.W_rand = torch.randn(dim, M)
        self.Q = torch.zeros(M, num_classes)
        self.G = torch.zeros(M, M)
        self.fc = CosineLinear(M, num_classes)


    def random_projection(self, features): # assume features are precomputed
        features = F.relu(features @ self.W_rand)
        return features
    
    def forward(self, features):
        features = self.random_projection(features)
        logits = self.fc(features)
        return logits
    
    def update_classes(self, num_classes):
        # update fc
        old_weight = self.fc.weight.data
        new_fc = CosineLinear(self.fc.in_features, num_classes)
        with torch.no_grad():
            new_fc.weight.data[:old_weight.size(0), :] = old_weight
        self.fc = new_fc

        # update Q
        old_Q = self.Q
        new_Q = torch.zeros(self.Q.size(0), num_classes)
        with torch.no_grad():
            new_Q[:, :old_Q.size(1)] = old_Q
        self.Q = new_Q


    def update_QG(self, features, labels):
        self.Q += features.T @ labels # requires labels to be one-hot encoded
        self.G += features.T @ features
        ridge = self.optimise_ridge_parameter(features, labels)
        Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T #better nmerical stability than .inv
        self.fc.weight.data = Wo
    
    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in tqdm(ridges):
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge
    
    def evaluate(self, test_features, test_labels):
        test_features = self.random_projection(test_features)
        logits = self.fc(test_features)
        _, preds = torch.max(logits, 1)
        acc = (preds == test_labels).float().mean()
        return acc
    


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


if __name__=="__main__":
    args = parse_args()
    assert args.num_jobs == 1, "Only one job is supported for RanPAC"
    model, transform, model_name, device = get_model(args)
    train_years, test_years = get_years(args)  
    results = {}
    os.makedirs("features", exist_ok=True)
    os.makedirs(f"features/{model_name}", exist_ok=True)

    class_order = ClassOrder()
    ranpac_model = RanPAC(model, args.M, len(class_order))
    num_classes = len(class_order)

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
            # update the class order
            unique_labels = set(train_labels)
            class_order.update(unique_labels)
            num_classes = len(class_order)
            ranpac_model.update_classes(num_classes)

            train_labels = torch.tensor([class_order.class_to_index(label) for label in train_labels])
            train_labels = target2onehot(train_labels, num_classes)

            train_embeddings = ranpac_model.random_projection(train_embeddings)
            ranpac_model.update_QG(train_embeddings, train_labels)
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
                test_labels = torch.tensor([class_order.class_to_index(label) for label in test_labels])
                #test_labels = target2onehot(test_labels, num_classes)

                acc = ranpac_model.evaluate(test_embeddings, test_labels)
                results[train_year][test_year] = float(acc)
                print(f"Train year: {train_year}, Test year: {test_year}, NCM accuracy: {acc:.4f}")
                pbar.update(1)
    # Save results
    os.makedirs("ranpac_results", exist_ok=True)
    os.makedirs(f"ranpac_results/{model_name}", exist_ok=True)
    with open(f"ranpac_results/{model_name}/ranpac_results_{args.job_id}.json", "w") as f:
        json.dump(results, f, indent=4)