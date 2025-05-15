<h1 align="center">CaMiT: Time-Aware Generation</h1>

## Stable Diffusion LoRA Fine-Tuning

This script fine-tunes a Stable Diffusion model using LoRA adapters with a dataset of car images and captions provided as a JSON file.

#### Generate Image-Caption Pairs
To create the image-caption JSON files needed for Stable Diffusion training, run:

```bash
python caption_generation.py --annotation_file /path/to/annotations.json --base_images_dir /path/to/images/
```

This will generate two files:

  * ```caption_pairs_without_year.json```

  * ```caption_pairs_with_year.json```

These contain image paths and corresponding captions formatted for training.

#### Installing the dependencies
Before running the training script, make sure to install the library's training dependencies:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
Then cd in the example folder and run
```bash
pip install -r requirements.txt
```
And initialize an 🤗 Accelerate environment with:
```bash
accelerate config
```
### How to Run

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
  --output_dir="/path/to/checkpoints/" \
  --rank=64 \
  --validation_prompt="some text to validate your training" \
  --validation_epochs=1 \
  --report_to="wandb"
```

## Image Generation Inference

This script generates car images using multiple Stable Diffusion models

#### Generating Captions for Inference

Run this script to create JSON files with prompts that will be used for inference. 
```bash
python inference_caption_generation.py --base_images_dir /path/to/test_images

```
This process creates two JSON files:

  * ```caption_counts_without_year.json``` — prompts ignoring year

  * ```caption_counts_with_year.json``` — prompts including year information for finer control

### How to Run
Launch training with:

```bash
python inference.py \
  --plain_model_path /path/to/plain_stable_diffusion \
  --finetuned_model_1 /path/to/lora_finetuned_model_1 \
  --finetuned_model_2 /path/to/lora_finetuned_model_2 \
  --caption_file_without_year /path/to/captions_without_year.json \
  --caption_file_with_year /path/to/captions_with_year.json \
  --model_choice <model_name> \
  --year_option <scenario> \
  --device cuda \
  --num_inference_steps 30 \
  --batch_size 4 \
  --seed 42
```

### Important Flags

```--caption_file_with_year``` and ```--caption_file_without_year```: JSON files containing with and without year prompts with image counts.

```--year_option``` "with_year" flag uses prompts including the year, "without_year" flag prompt ignores year info.

```--model_choice```: Select which model to run inference with:
```bash
      "plain_sd" — base Stable Diffusion

      "finetuned1" — first LoRA fine-tuned model

      "finetuned2" — second LoRA fine-tuned model
```
## Embedding Extraction for Generated and Real Images

Run the script specifying the model, generation method, scenario, and data directories:

```bash
python extract_embeddings.py \
  <model_name> <method> <scenario> \
  --gen_root /path/to/generated_images \
  --real_root /path/to/real_images \
  --emb_root /path/to/save_embeddings \
  --world_size 4
```

* ```model_name```: The model variant for embeddings extraction (clip_base, clip_large etc ...)
* ```method```: Choose a generation method (plain_sd,finetuned1, finetuned2)
* ```scenario```: Choose a caption format either ```"with_year"``` (prompts with car models + year) or ```"without_year"``` (prompts car_models)
* ```world_size```: Number of GPUs/processes used for parallel embedding extraction.

  
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
