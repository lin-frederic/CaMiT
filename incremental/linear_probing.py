import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import SupervisedTimeDataset, ClassOrder
from models import get_model, LinearProbingModel, get_model_dim, load_embeddings
from utils import parse_args, get_years, get_collate_fn

class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

if __name__ == "__main__":
    args = parse_args()
    assert args.num_jobs == 1, "Only one job is supported for linear probing"
    model, transform, model_name, device = get_model(args)

    train_years, test_years = get_years(args)
    results = {}

    class_order = ClassOrder()

    model_dim = get_model_dim(model)
    num_classes = len(class_order)

    linear_model = None

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

            unique_labels = set(train_labels)
            class_order.update(unique_labels)
            num_classes = len(class_order)

            train_labels = torch.tensor([class_order.class_to_index(label) for label in train_labels])
            if linear_model is None:
                linear_model = LinearProbingModel(output_dim=model_dim, num_classes=num_classes)
                linear_model.to(device)
            else:
                linear_model.update_classifier(num_classes=num_classes)
                linear_model.to(device)

            train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            optimizer = torch.optim.SGD(linear_model.parameters(), lr=args.linear_probing_lr, momentum=0.9, weight_decay=4e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*args.linear_probing_epochs, eta_min=0)
            criterion = nn.CrossEntropyLoss()
            linear_model.train()

            for epoch in range(args.linear_probing_epochs):
                for i, (embeddings, labels) in enumerate(train_dataloader):
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = linear_model(embeddings)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if i % 100 == 0:
                        print(f"Epoch [{epoch+1}/{args.linear_probing_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            linear_model.eval()
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

                test_dataset = EmbeddingsDataset(test_embeddings, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                correct = 0
                total = 0

                with torch.no_grad():
                    for embeddings, labels in test_dataloader:
                        embeddings = embeddings.to(device)
                        labels = labels.to(device)

                        outputs = linear_model(embeddings)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                acc = correct / total
                results[train_year][test_year] = float(acc)
                print(f"Train year: {train_year}, Test year: {test_year}, Linear Probing accuracy: {acc:.4f}")
                pbar.update(1)

    # Save results
    os.makedirs("linear_results", exist_ok=True)
    os.makedirs(f"linear_results/{model_name}", exist_ok=True)
    with open(f"linear_results/{model_name}/linear_results.json", "w") as f:
        json.dump(results, f, indent=4)

            





    