import os
import torch
import torchvision.transforms as transforms
from dataset_cropvlm import CropVLM, resize_transform
import argparse
import math
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    epoch_loss = 0.
    epoch_acc1 = 0.
    epoch_acc5 = 0.
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        output = model(images)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        epoch_acc1 += acc1.item()
        epoch_acc5 += acc5.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to TensorBoard every 10 batches
        if i % 1 == 0: 
            print(f"Epoch: {epoch} [{i}/{len(train_loader)}] Loss: {loss.item()} Acc@1: {acc1.item()} Acc@5: {acc5.item()}")
            
            # Log metrics to TensorBoard
            writer.add_scalar('Train/Loss', epoch_loss / (i + 1), epoch * len(train_loader) + i) # Average loss per batch
            writer.add_scalar('Train/Acc@1', epoch_acc1 / (i + 1), epoch * len(train_loader) + i) # Average top-1 accuracy per batch
            writer.add_scalar('Train/Acc@5', epoch_acc5 / (i + 1), epoch * len(train_loader) + i)

def validate(val_loader, model, criterion, args, writer, epoch):
    model.eval()    
    avg_acc1 = 0.
    avg_acc5 = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Log validation metrics to TensorBoard
            if i % 10 == 0:
                print(f"Validation [{i}/{len(val_loader)}] Loss: {loss.item()} Acc@1: {acc1.item()} Acc@5: {acc5.item()}")

            avg_acc1 += acc1.item()
            avg_acc5 += acc5.item()

    avg_acc1 /= len(val_loader)
    avg_acc5 /= len(val_loader)
    print(f"Validation Acc@1: {avg_acc1} Acc@5: {avg_acc5}")

    # Log final validation accuracy to TensorBoard
    writer.add_scalar('Validation/Acc@1', avg_acc1, epoch)
    writer.add_scalar('Validation/Acc@5', avg_acc5, epoch)

    return avg_acc1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_annotations_path", type=str, default="outputs/object_annotations.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=90)
    
    args = parser.parse_args()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/linear_probing_dinov2")

    # load dataset
    dataset = CropVLM(args.object_annotations_path)
    dataset.to_supervised()

    n_classes = len(dataset.brand_to_idx)
    print(f"Number of classes: {n_classes}")

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset.dataset.transform = transforms.Compose([
        transforms.Lambda(resize_transform),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset.dataset.transform = transforms.Compose([
        transforms.Lambda(resize_transform),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load dinov2 model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)

    # replace head
    model.head = torch.nn.Linear(model.embed_dim, n_classes)

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    init_lr = args.lr * args.batch_size / 256

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        raise NotImplementedError("No GPU available")

    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters())) # only train head
    assert len(parameters) == 2, "Only head should be trainable"
    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=0.9, weight_decay=0.)

    best_acc1 = 0.
    os.makedirs("linear_probing_dinov2", exist_ok=True)
    print("Start training")
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        acc1 = validate(val_loader, model, criterion, args, writer, epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_file = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }
        # save checkpoint
        torch.save(save_file , f'linear_probing_dinov2/checkpoint_{epoch}.pth.tar')

        if is_best:
            torch.save(save_file, f'linear_probing_dinov2/best.pth.tar')

    # Close the writer after training is done
    writer.close()
