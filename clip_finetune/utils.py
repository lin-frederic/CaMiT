import os
import argparse
import torch

import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotations", type=str, help="Path to train annotations JSON file", default="cars_dataset/train_annotations.json")
    parser.add_argument("--train_images_dir", type=str, help="Path to train images directory", default="cars_dataset/train_blurred")
    parser.add_argument("--test_annotations", type=str, help="Path to test annotations JSON file", default="cars_dataset/test_annotations.json")
    parser.add_argument("--test_images_dir", type=str, help="Path to test images directory", default="cars_dataset/test_blurred")

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")

    parser.add_argument("--model_size", type=str, default="b", help="Model size (s, b,l)")
    parser.add_argument("--model_path", type=str, default="lora_clip/checkpoints_b/checkpoint_19.pth", help="Path to the model checkpoint")
    parser.add_argument("--pretraining_time", type=str, default="2007", help="Pretraining time of the model")

    parser.add_argument("--job_id", type=int, default=0, help="Job ID for parallel processing")
    parser.add_argument("--num_jobs", type=int, default=8, help="Number of jobs for parallel processing")

    parser.add_argument("--alpha1", type=float, default=1, help="Alpha1 for covariance computation")
    parser.add_argument("--alpha2", type=float, default=0, help="Alpha2 for covariance computation")

    # RanPAC specific arguments
    parser.add_argument("--M", type=int, default=10000, help="Dimension of the projected space")
    
    # linear probing specific arguments
    parser.add_argument("--linear_probing_epochs", type=int, default=300, help="Number of epochs for linear probing")
    parser.add_argument("--linear_probing_lr", type=float, default=0.1, help="Learning rate for linear probing")

    args = parser.parse_args()



    return args

def get_years(args):
    train_years = sorted(os.listdir(args.train_images_dir), key=lambda x: int(x)) # 2007-2023: 17 years
    num_train_years = len(train_years)
    # Split train_years into num_jobs parts
    num_train_years_per_job = num_train_years // args.num_jobs
    start_index = args.job_id * num_train_years_per_job
    end_index = (args.job_id + 1) * num_train_years_per_job if args.job_id != args.num_jobs - 1 else num_train_years
    train_years = train_years[start_index:end_index]
    print(f"Job {args.job_id}: Processing train years {train_years}")
    test_years = sorted(os.listdir(args.test_images_dir), key=lambda x: int(x))
    return train_years, test_years

def collate_fn_qwen(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

def onehot2target(onehot):
    return torch.argmax(onehot, dim=1)

def set_seed():
    random_seed = random.randint(0, 10000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return random_seed
