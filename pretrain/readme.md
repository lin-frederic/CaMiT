# 🔧 Pretraining (MoCo v3 with ViT-S)

This folder contains scripts for self-supervised pretraining of a vision transformer (ViT-S) encoder using the [MoCo v3](https://github.com/facebookresearch/moco) framework. The encoder is trained on filtered car crops to learn a car-specific representation, which is later used for fine-tuning.

## 🗂️ Data

- `--pretrain_data`: Path to unlabeled car crops for self-supervised training.
- `--train_data`: Path to weakly labeled car crops used during training (the labels are not used for pretraining).
## 🚀 Launch

Example command:

python pretrain/initial_pretrain.py \
  -a vit_small -b 2048 -j 32 \
  --pretrain_data /path/to/pretrain_data \
  --train_data /path/to/train_data \
  --optimizer adamw --lr 1.5e-4 --weight-decay 0.1 \
  --epochs 300 --warmup-epochs 40 \
  --stop-grad-conv1 --moco-m-cos --moco-t 0.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --logdir initial_pretrain_with_train
