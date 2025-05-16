import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import matplotlib.pyplot as plt
import random
from dataset_cropvlm import CropVLM, resize_transform
import sys
moco_path = os.path.join(os.path.dirname(__file__), "moco-v3")
if moco_path not in sys.path:
    sys.path.append(moco_path)
    sys.path.append(os.path.join(moco_path, "moco"))
import vits  # from moco-v3
from tqdm import tqdm  # Progress bar

def load_model(args):
    print(f"=> creating model '{args.arch}'")
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch]()
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch]()
        linear_keyword = 'fc'

    model.reset_classifier(num_classes=89)  # 89 classes in CropVLM

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"=> loading checkpoint '{args.checkpoint}'")
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            # remove prefix 'module.' for dataparallel
            if 'state_dict' in checkpoint:
                for k in list(checkpoint['state_dict'].keys()):
                    if k.startswith('module.'):
                        checkpoint['state_dict'][k[len('module.'):]] = checkpoint['state_dict'].pop(k) 
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.checkpoint}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.checkpoint}', exiting.")
            exit(1)

    return model

def evaluate(model, val_loader, device, num_classes):
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    model.eval()

    correct_per_class = [0]*num_classes
    total_per_class = [0]*num_classes
    top1_correct = 0
    total_samples = 0
    loss_sum = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", unit="batch")
        for images, target in progress_bar:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)

            loss_sum += loss.item() * images.size(0)
            _, pred = output.topk(1, 1, True, True)
            top1_correct += pred.eq(target.view(-1, 1)).sum().item()
            total_samples += images.size(0)

            # Calculate per-class accuracy
            for t,p in zip(target, pred.view(-1)):
                total_per_class[t.item()] += 1
                if t.item() == p.item():
                    correct_per_class[t.item()] += 1


            avg_loss = loss_sum / total_samples
            top1_acc = 100.0 * top1_correct / total_samples

            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{top1_acc:.2f}%")

    print(f"Final Validation Loss: {avg_loss:.4f}, Accuracy: {top1_acc:.2f}%")
    class_accuracies = [100.0*correct/total if total > 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]
    return class_accuracies

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--object_annotations_path", type=str, default="outputs/object_annotations.json", help="Path to object annotations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--arch", type=str, default="vit_base", help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=8, help="Number of data workers")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = load_model(args)

    transform = transforms.Compose([
        transforms.Lambda(resize_transform),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CropVLM(args.object_annotations_path)
    dataset.to_supervised()
    dataset.transform = transform

    # limit the number of samples to evaluate
    total_samples = len(dataset)
    num_classes = len(dataset.brand_to_idx)
    class_names = list(dataset.brand_to_idx.keys())
    #sample_indices = random.sample(range(total_samples), min(10000, total_samples)) # might not sample all classes
    #dataset = torch.utils.data.Subset(dataset, sample_indices)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(f"Evaluating on {len(dataset)} samples...")
    class_accuracies = evaluate(model, val_loader, device, num_classes)
    # Sort by accuracy
    class_names, class_accuracies = zip(*sorted(zip(class_names, class_accuracies), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(15, 20))
    plt.barh(class_names, class_accuracies, color='skyblue')
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Class Name")
    plt.title("Per-Class Accuracy")
    plt.savefig("outputs/moco_per_class_accuracy.png")

if __name__ == "__main__":
    main()
