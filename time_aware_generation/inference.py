import os
import json
import argparse
import torch
from diffusers import DiffusionPipeline
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import time
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model, multi-caption car image generator.")
    parser.add_argument("--plain_model_path", type=str, required=True)
    parser.add_argument("--finetuned_model_1", type=str, required=True)
    parser.add_argument("--finetuned_model_2", type=str, required=True)
    parser.add_argument("--caption_file_without_year", type=str, required=True)
    parser.add_argument("--caption_file_with_year", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="generated_cars")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--model_choice",
        type=str,
        choices=["plain_sd", "finetuned1", "finetuned2"],
        required=True,
        help="Choose which model to run inference with."
    )
    parser.add_argument(
        "--year_option",
        type=str,
        choices=["with_year", "without_year"],
        required=True,
        help="Choose whether to use captions with year or without year."
    )

    return parser.parse_args()

def load_pipe(model_path, device, lora_path=None):
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    pipe = pipe.to(device)

    if lora_path is not None:
        pipe.load_lora_weights(lora_path)

    return pipe

def generate_images(pipe, prompt_data, save_base_dir, num_inference_steps, device, seed=None, with_year=False, batch_size=4):
    total_start = time.time()
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    os.makedirs(save_base_dir, exist_ok=True)
    for item in tqdm(prompt_data, desc="Prompts"):
        prompt = item["prompt"]
        count = item["count"]

        prompt_start = time.time()

        prefix = "A photo of"
        if not prompt.startswith(prefix):
            raise ValueError(f"Prompt format unexpected: {prompt}")
        
        after_prefix = prompt[len(prefix):]

        if with_year:
            model_name, _, year = after_prefix.rpartition(" in ")
            model_name = model_name.strip()
            year = year.strip()
            model_dir = model_name.replace(' ', '_')
            save_dir = os.path.join(save_base_dir, model_dir, year)
        else:
            model_name = after_prefix.strip()
            model_dir = model_name.replace(' ', '_')
            save_dir = os.path.join(save_base_dir, model_dir)

        os.makedirs(save_dir, exist_ok=True)

        n_batches = math.ceil(count / batch_size)

        for b in tqdm(range(n_batches), desc=f"Batching for '{model_dir}'", leave=False):
            current_batch_size = min(batch_size, count - b * batch_size)
            prompts_batch = [prompt] * current_batch_size

            with torch.autocast(device):
                images = pipe(
                    prompt=prompts_batch,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images

            for i, image in enumerate(images):
                idx = b * batch_size + i
                image.save(os.path.join(save_dir, f"{model_name.replace(' ', '_')}_{idx}.png"))

        prompt_time = time.time() - prompt_start
        print(f"Finished '{prompt}' with {count} images in {prompt_time:.2f} seconds.")

    total_time = time.time() - total_start
    print(f"\n Total generation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

def main():
    args = parse_args()
    if args.year_option == "with_year":
        with open(args.caption_file_with_year, 'r') as f:
            captions = json.load(f)
        with_year_flag = True
    else:
        with open(args.caption_file_without_year, 'r') as f:
            captions = json.load(f)
        with_year_flag = False

    if args.model_choice == "plain_sd":
        print("Loading Plain SD Model...")
        pipe = load_pipe(args.plain_model_path, args.device)
    elif args.model_choice == "finetuned1":
        print("Loading Fine-tuned Model 1...")
        pipe = load_pipe(args.plain_model_path, args.device, lora_path=args.finetuned_model_1)
    elif args.model_choice == "finetuned2": 
        print("Loading Fine-tuned Model 2...")
        pipe = load_pipe(args.plain_model_path, args.device, lora_path=args.finetuned_model_2)


    save_dir = os.path.join(args.output_dir, args.model_choice, args.year_option)
    generate_images(
        pipe=pipe,
        prompt_data=captions,
        save_base_dir=save_dir,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        seed=args.seed,
        with_year=with_year_flag,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
