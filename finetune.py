from copy import deepcopy
import datetime
from glob import glob
import inspect
import logging
import math
import os
from typing import Optional, Union, List

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import argparse
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

# Models and Pipelines
from midi.models.transformers import TripoSGDiTModel
from midi.pipelines.pipeline_midi import MIDIPipeline
from midi.utils.smoothing import smooth_gpu # for post processing in marching cubes

# Dataset
from dataset import SyntheticDataset

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def get_model():
    local_dir = "pretrained_weights/MIDI-3D/transformer"
    sparse_structure_flow_model = TripoSGDiTModel.from_pretrained(local_dir)
    return sparse_structure_flow_model

def create_output_folders(output_dir, config, exp_name):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"{exp_name}_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
    return out_dir

def sample_flow_matching_t_for_training(logit_normal_mu, logit_normal_sigma, bsz):
    """
    Sample t for flow matching training with a LogitNormal distribution.
    """
    t = torch.randn(bsz)
    t = torch.sigmoid(logit_normal_mu + t * logit_normal_sigma)
    return t

def forward_flow_matching_loss(model, x0, sigma, cond, attention_kwargs, eps=None, **kwargs):
    if eps is None:
        eps = torch.randn_like(x0).to(x0.device)
    # Reshape sigma for broadcasting
    sigma_reshaped = sigma.view(sigma.shape + (1,) * (x0.ndim - 1))
    
    # Create noisy samples using SIGMA (not timestep)
    xt = (1 - sigma_reshaped) * x0 + sigma_reshaped * eps
    
    # Convert sigma to timestep for model input
    # timestep = sigma * num_train_timesteps  # [B] -> [0, 1000] range
    timestep = (sigma * 1000).to(dtype=x0.dtype) 
    
    # Target is the velocity field
    target = eps - x0
    
    # Model prediction - use POSITIONAL arguments for latent and timestep
    pred = model(
        xt,                                      # positional: latent_model_input
        timestep,                                # positional: timestep in [0, 1000]
        encoder_hidden_states=cond['image_embeds_1'].to(x0.device, dtype=x0.dtype),
        encoder_hidden_states_2=cond['image_embeds_2'].to(x0.device, dtype=x0.dtype),
        attention_kwargs=attention_kwargs,
        return_dict=False
    )[0]
    
    # Compute loss
    loss = (pred - target).pow(2).mean(dim=tuple(range(1, x0.ndim)))
    return loss


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    model = get_model()
    model.to(device, dtype=dtype)

    train_dataset = SyntheticDataset(
        root_dir='./synthetic_data',
        category='category',
        num_models_per_image=4,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2
    )

    for i, batch in enumerate(train_loader):
        x0 = batch['model_win_sparse_x0'].to(device, dtype=dtype)
        x0 = x0.squeeze(1)
        # Sample sigma (noise level) in [0, 1]
        sigma = sample_flow_matching_t_for_training(
            logit_normal_mu=0.0, 
            logit_normal_sigma=1.0, 
            bsz=2
        ).to(device, dtype=dtype) 

        # Prepare attention kwargs
        attention_kwargs={
            "num_instances": 1,
        }

        # Compute loss
        loss = forward_flow_matching_loss(
            model=model,
            x0=x0,
            sigma=sigma,  # NOT t, and NOT t*1000!
            cond=batch['cond'],
            attention_kwargs=attention_kwargs
        )
        print(loss)
        