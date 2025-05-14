
# Stable Diffusion LoRA Fine-Tuning

This script fine-tunes a Stable Diffusion model using LoRA adapters on a custom dataset of car images and captions.

## How to Run

Launch training with:

```bash
export WANDB_MODE=offline
export MODEL_NAME="/path/to/stable-diffusion-v1-5"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --caption_json="/path/to/caption_pairs.json" \
  --base_images_dir="/path/to/images_directory/" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=30 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-4 \
  --resume_from_checkpoint="latest" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --dataloader_num_workers=6 \
  --seed=42 \
  --output_dir="sd-car-model-lora" \
  --rank=64 \
  --validation_prompt="A photo of a car bentley continental" \
  --validation_epochs=1 \
  --report_to="wandb"
```

## KID Computation
This script computes the Kernel Inception Distance (KID) between real and generated image embeddings for different models and scenarios.
## How to Run
```bash
python compute_kid.py MODEL_NAME --emb_root /path/to/embeddings
```

* ```MODEL_NAME```: The model variant whose embeddings will be loaded for KID calculation.
* ```--emb_root```: Root directory where embeddings are stored

## Example
```bash
python compute_kid.py clip_base --emb_root /home/data/bambaw/cars_finetune/embeddings
```
The script computes KID scores and saves results and plots in the current directory.
