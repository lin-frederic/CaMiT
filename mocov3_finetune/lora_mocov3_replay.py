import os
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataset import SupervisedTimeDatasetWithReplay
from models import get_model_with_lora

from torch.utils.tensorboard import SummaryWriter

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.sigma.data.fill_(1)
    
    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        out = out * self.sigma
        return out

class ModelWithHead(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ModelWithHead, self).__init__()
        self.base_model = base_model
        self.base_model.head = nn.Identity()  # Remove the original head
        self.base_model.to("cuda")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.base_model(dummy_input.to("cuda"))
        dim = output.shape[1]
        self.head = CosineLinear(dim, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.head(x)
        return x

def train_epoch(model, train_loader, optimizer, scheduler, writer, epoch):
    model.train()
    losses = 0.0
    correct, total = 0, 0
    # add tqdm progress bar
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        if i % 10 == 0:
            step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Accuracy", correct / total * 100, step)
            print(f"Step {step}: Loss: {loss.item():.4f}, Accuracy: {correct / total * 100:.2f}%")

    scheduler.step()
    train_acc = round(correct / total * 100, 2)
    train_loss = losses / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    return train_loss, train_acc

def test_epoch(model, test_loader, writer, epoch):
    model.eval()
    correct, total = 0, 0 
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_acc = round(correct / total * 100, 2)
    print(f"Test Accuracy: {test_acc:.4f}")
    writer.add_scalar("Test/Accuracy", test_acc, epoch)


    return test_acc


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to train annotations JSON file", default="cars_dataset/train_annotations.json")
    parser.add_argument("--train_images_dir", type=str, help="Path to train images directory", default="cars_dataset/train_blurred")
    parser.add_argument("--test_annotations", type=str, help="Path to test annotations JSON file", default="cars_dataset/test_annotations.json")
    parser.add_argument("--test_images_dir", type=str, help="Path to test images directory", default="cars_dataset/test_blurred")

    parser.add_argument("--model_path", type=str, default="update_2008/checkpoints/checkpoint_0259.pth.tar", help="Path to pre-trained model checkpoint")
    parser.add_argument("--model_size", type=str, default="s", help="Model size: s or b")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training and testing") # suboptimal (using only 4GB of GPU memory) but same one as in RanPAC
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for SGD optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD optimizer")

    parser.add_argument("--pretraining_year", type=str, default="2008", help="Year the pre-trained model was trained on")


    parser.add_argument("--save_dir", type=str, default="lora_mocov3/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="lora_mocov3/runs", help="Directory to save logs")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_size = args.model_size
    model_path = args.model_path
    model = get_model_with_lora(model_size, model_path)

    
    preprocess_train = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), # same as in CLIP 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    preprocess_val = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = SupervisedTimeDatasetWithReplay(args.train_annotations, args.train_images_dir, args.pretraining_year , transform=preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = SupervisedTimeDatasetWithReplay(args.test_annotations, args.test_images_dir, args.pretraining_year , transform=preprocess_val)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(test_dataset.class_to_idx)
    
    model = ModelWithHead(model, num_classes)

    model = model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters (with head): {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.SGD(trainable_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, writer, epoch)
        test_acc = test_epoch(model, test_dataloader, writer, epoch)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint_{args.pretraining_year}_replay_{epoch}.pth"))



