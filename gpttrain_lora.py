"""
Training script for finetuning TripoSGDiTModel with LoRA
This script demonstrates how to finetune the MIDI-3D transformer using LoRA for 3D generation.

Usage:
    python train_lora.py --config configs/train/lora_finetune.yaml
    
Or with command line arguments:
    python train_lora.py --pretrained_model pretrained_weights/MIDI-3D \
                          --output_dir outputs/lora_finetune \
                          --batch_size 4 \
                          --num_epochs 100
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from midi.inference_utils import generate_dense_grid_points
from midi.models.autoencoders import TripoSGVAEModel
from midi.models.transformers import TripoSGDiTModel
from midi.pipelines.pipeline_midi import MIDIPipeline
from midi.schedulers import (
    RectifiedFlowScheduler,
    compute_density_for_timestep_sampling,
    compute_loss_weighting,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA finetuning for MIDI-3D Transformer")
    
    # Model paths
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained_weights/MIDI-3D",
        help="Path to pretrained MIDI-3D model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_finetune",
        help="Output directory for checkpoints and logs"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (r parameter)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="Target modules for LoRA"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for training"
    )
    
    # Training noise scheduler
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal_dist",
        choices=["sigma_sqrt", "logit_normal", "logit_normal_dist", "mode", "cosmap"],
        help="Weighting scheme for loss"
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean for logit normal distribution"
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std for logit normal distribution"
    )
    
    # Checkpointing
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    
    # Multi-instance attention
    parser.add_argument(
        "--enable_multi_instance_attention",
        action="store_true",
        default=True,
        help="Enable multi-instance attention processor"
    )
    parser.add_argument(
        "--mi_attention_blocks",
        type=str,
        nargs="+",
        default=["blocks.8", "blocks.9", "blocks.10", "blocks.11", "blocks.12"],
        help="Transformer blocks to apply multi-instance attention"
    )
    
    # Additional options
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    return args


class MIDI3DDataset(Dataset):
    """
    Custom dataset for MIDI-3D training.
    Expects data directory with structure:
        data_dir/
            scene_0001/
                rgb.png          # RGB image
                seg.png          # Segmentation mask
                scene_rgb.png    # Scene RGB
                surface.npy      # Ground truth surface points [N, 3] or [N, 4] with normals
            scene_0002/
                ...
    """
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Find all scene directories
        self.scenes = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(self.scenes) == 0:
            raise ValueError(f"No scene directories found in {data_dir}")
        
        logger.info(f"Found {len(self.scenes)} scenes in {data_dir}")
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene_dir = self.scenes[idx]
        
        # Load images
        rgb_path = scene_dir / "rgb.png"
        seg_path = scene_dir / "seg.png"
        scene_rgb_path = scene_dir / "scene_rgb.png"
        surface_path = scene_dir / "surface.npy"
        
        # Load and preprocess images
        rgb = Image.open(rgb_path).convert("RGB")
        seg = Image.open(seg_path).convert("L")
        scene_rgb = Image.open(scene_rgb_path).convert("RGB")
        
        # Resize if needed
        if rgb.size != (self.image_size, self.image_size):
            rgb = rgb.resize((self.image_size, self.image_size), Image.LANCZOS)
            seg = seg.resize((self.image_size, self.image_size), Image.NEAREST)
            scene_rgb = scene_rgb.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensors
        rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        seg = torch.from_numpy(np.array(seg)).unsqueeze(0).float() / 255.0
        scene_rgb = torch.from_numpy(np.array(scene_rgb)).permute(2, 0, 1).float() / 255.0
        
        # Load surface points
        surface = np.load(surface_path)
        surface = torch.from_numpy(surface).float()
        
        # Calculate number of instances from segmentation
        seg_np = (seg.squeeze().numpy() * 255).astype(np.uint8)
        unique_labels = np.unique(seg_np)
        num_instances = len(unique_labels[unique_labels > 0])
        
        return {
            "rgb": rgb,
            "seg": seg,
            "scene_rgb": scene_rgb,
            "surface": surface,
            "num_instances": num_instances,
            "scene_name": scene_dir.name
        }


def collate_fn(batch):
    """Custom collate function to handle variable length data"""
    # Stack images
    rgb = torch.stack([item["rgb"] for item in batch])
    seg = torch.stack([item["seg"] for item in batch])
    scene_rgb = torch.stack([item["scene_rgb"] for item in batch])
    
    # Handle variable-length surface points by padding
    max_points = max([item["surface"].shape[0] for item in batch])
    surface_list = []
    for item in batch:
        surf = item["surface"]
        if surf.shape[0] < max_points:
            # Pad with zeros
            padding = torch.zeros(max_points - surf.shape[0], surf.shape[1])
            surf = torch.cat([surf, padding], dim=0)
        surface_list.append(surf)
    
    surface = torch.stack(surface_list)
    num_instances = torch.tensor([item["num_instances"] for item in batch])
    
    return {
        "rgb": rgb,
        "seg": seg,
        "scene_rgb": scene_rgb,
        "surface": surface,
        "num_instances": num_instances,
        "num_instances_per_batch": num_instances,
    }


def encode_vae(vae, surface_points):
    """Encode surface points to latent space using VAE encoder"""
    with torch.no_grad():
        latent_dist = vae.encode(surface_points)
        latents = latent_dist.sample()
    return latents


def compute_loss(model_pred, target, weighting, reduction="mean"):
    """Compute weighted MSE loss"""
    loss = F.mse_loss(model_pred, target, reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    loss = loss * weighting
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load pretrained pipeline
    logger.info(f"Loading pretrained model from {args.pretrained_model}")
    pipeline: MIDIPipeline = MIDIPipeline.from_pretrained(args.pretrained_model)
    
    # Initialize custom adapter with LoRA
    transformer_lora_config = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.lora_target_modules,
        "lora_dropout": args.lora_dropout,
    }
    
    logger.info(f"Initializing LoRA with config: {transformer_lora_config}")
    pipeline.init_custom_adapter(
        set_self_attn_module_names=args.mi_attention_blocks if args.enable_multi_instance_attention else None,
        transformer_lora_config=transformer_lora_config,
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing()
    
    # Extract components
    vae: TripoSGVAEModel = pipeline.vae
    transformer: TripoSGDiTModel = pipeline.transformer
    image_encoder_1: CLIPVisionModelWithProjection = pipeline.image_encoder_1
    image_encoder_2: Dinov2Model = pipeline.image_encoder_2
    noise_scheduler: RectifiedFlowScheduler = RectifiedFlowScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Freeze components (only train LoRA parameters)
    vae.requires_grad_(False)
    image_encoder_1.requires_grad_(False)
    image_encoder_2.requires_grad_(False)
    
    # Only LoRA parameters in transformer will have requires_grad=True
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create dataset and dataloader
    logger.info(f"Loading dataset from {args.data_dir}")
    train_dataset = MIDI3DDataset(args.data_dir, image_size=args.image_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )
    
    # Prepare with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move other components to device
    vae = vae.to(accelerator.device)
    image_encoder_1 = image_encoder_1.to(accelerator.device)
    image_encoder_2 = image_encoder_2.to(accelerator.device)
    
    # Training info
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Training loop
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # Extract epoch from checkpoint path if named like "checkpoint-epoch-10"
        if "epoch" in args.resume_from_checkpoint:
            first_epoch = int(args.resume_from_checkpoint.split("-")[-1])
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_epochs):
        transformer.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode images with CLIP and DINOv2
                with torch.no_grad():
                    # Encode with image_encoder_1 (CLIP)
                    image_embeds_1 = pipeline.encode_image_1(
                        batch["rgb"],
                        accelerator.device,
                        num_images_per_prompt=1
                    )[0]  # Use conditional embeddings only
                    
                    # Encode with image_encoder_2 (DINOv2)
                    image_embeds_2 = pipeline.encode_image_2(
                        batch["rgb"],
                        batch["scene_rgb"],
                        batch["seg"],
                        accelerator.device,
                        num_images_per_prompt=1
                    )[0]  # Use conditional embeddings only
                    
                    # Encode surface points to latent space
                    # For training, we need ground truth latents
                    # Note: This assumes you have a way to get GT latents
                    # In practice, you might need to encode GT 3D shapes
                    latents = encode_vae(vae, batch["surface"])
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=accelerator.device)
                
                # Add noise to latents (forward diffusion)
                sigmas = timesteps / noise_scheduler.config.num_train_timesteps
                sigmas = sigmas.view(-1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise
                
                # Predict the noise residual
                model_pred = transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=image_embeds_1,
                    encoder_hidden_states_2=image_embeds_2,
                    attention_kwargs={
                        "num_instances": batch["num_instances"],
                        "num_instances_per_batch": batch["num_instances_per_batch"],
                    },
                    return_dict=False,
                )[0]
                
                # Compute target (for flow matching, target is the velocity)
                target = noise - latents
                
                # Compute loss weighting
                weighting = compute_loss_weighting(args.weighting_scheme, sigmas.squeeze())
                
                # Compute loss
                loss = compute_loss(model_pred, target, weighting)
                
                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagation
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
        # Save checkpoint at end of epoch
        if (epoch + 1) % args.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                accelerator.save_state(save_path)
                
                # Also save just the LoRA weights
                lora_save_path = os.path.join(args.output_dir, f"lora-epoch-{epoch + 1}")
                os.makedirs(lora_save_path, exist_ok=True)
                
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
                torch.save(lora_state_dict, os.path.join(lora_save_path, "lora_weights.pth"))
                
                logger.info(f"Saved LoRA weights to {lora_save_path}")
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoint-final")
        accelerator.save_state(save_path)
        
        # Save final LoRA weights
        lora_save_path = os.path.join(args.output_dir, "lora-final")
        os.makedirs(lora_save_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
        torch.save(lora_state_dict, os.path.join(lora_save_path, "lora_weights.pth"))
        
        # Save training args
        import json
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info(f"Training completed! Final model saved to {args.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
