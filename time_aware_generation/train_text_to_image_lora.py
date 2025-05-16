import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
import os
import json
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import AutoTokenizer 

from wandb import Settings 
if is_wandb_available():
    import wandb

os.environ["WANDB_MODE"] = "offline"

check_min_version("0.33.0.dev0")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CarCaptionDataset(Dataset):
    def __init__(self, json_file, base_dir, tokenizer, resolution=512, center_crop=False, random_flip=False, max_length=77):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bad_images_log = "bad_images_during_training.txt"

        transform_list = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        transform_list.append(transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.base_dir, item["image"])
        caption = item["caption"]

        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transform(image)
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids.squeeze(0)

            return {"pixel_values": pixel_values, "input_ids": input_ids}
        
        except Exception as e:
            with open(self.bad_images_log, "a") as f:
                f.write(f"{image_path} - {str(e)}\n")
            
            dummy_pixel_values = torch.zeros((3, 512, 512)) 
            dummy_input_ids = torch.zeros((self.max_length,), dtype=torch.long)

            return {"pixel_values": dummy_pixel_values, "input_ids": dummy_input_ids}

def log_validation( pipeline, args, accelerator, epoch,is_final_validation=False,):
    
    logger.info(
        f"Running validation.... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Stable Diffusion with a custom JSON dataset.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained model (e.g., CompVis/stable-diffusion-v1-4).")
    
 
    parser.add_argument("--caption_json", type=str, required=True,
                        help="Path to the JSON file with image–caption pairs.")
    parser.add_argument("--base_images_dir", type=str, required=True,
                        help="Base directory where image files are stored.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="Prompt for generating validation images.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images to generate during validation.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run validation every X epochs.")
   
    parser.add_argument("--noise_offset", type=float, default=0, help="Scale for noise offset.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Center crop the images.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps (overrides epochs if set).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of warmup steps for LR scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--checkpointing_steps", type=int, default=5000,help="Save a checkpoint every X training steps.")
    parser.add_argument("--output_dir", type=str, default="sd-car-model-lora", help="Output directory for checkpoints and final model.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging integration (tensorboard, wandb, etc.).")
    parser.add_argument("--push_to_hub", action="store_true", help="Push final model to the Hub.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."),)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),)
    

    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--rank", type=int, default=4, help="Dimension (rank) of LoRA update matrices.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading.")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if needed.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of model files (e.g., fp16).")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR gamma for loss rebalancing.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode.")
    parser.add_argument("--logging_dir",type=str,default="logs", help=( "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to" " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
   
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)


    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    accelerator.init_trackers(
        "text2image-fine-tune",
        config=vars(args),
        init_kwargs={"wandb": {"settings": Settings(init_timeout=180)}}
    )

    device = accelerator.device

 
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger.info("Accelerator state: %s", accelerator.state)
    
 

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant=args.variant)
    
   
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    from torch.utils.data import DataLoader
    dataset = CarCaptionDataset(
        json_file=args.caption_json,
        base_dir=args.base_images_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        max_length=77
    )
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.dataloader_num_workers ,pin_memory = True, persistent_workers=True,
prefetch_factor=2 )


    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,  
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    
    
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes via pip install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    
   
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:

            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
               
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

              
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

   
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        if args.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
 
            pipeline.load_lora_weights(args.output_dir)
            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)


    accelerator.end_training()


if __name__ == "__main__":
    main()