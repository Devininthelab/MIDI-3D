# LoRA Finetuning for MIDI-3D Transformer

This directory contains scripts and configurations for finetuning the TripoSGDiTModel (transformer) component of MIDI-3D using LoRA (Low-Rank Adaptation).

## Overview

LoRA is an efficient finetuning technique that adds trainable low-rank matrices to the transformer's attention layers, allowing you to adapt the model to your specific domain with minimal computational cost and storage requirements.

### Why LoRA for MIDI-3D?

- **Efficient**: Only trains ~1-5% of model parameters
- **Fast**: Significantly faster than full finetuning
- **Memory-friendly**: Lower GPU memory requirements
- **Portable**: LoRA weights are small (~50-200MB vs several GB for full model)
- **Reversible**: Can easily switch between different LoRA adaptations

## Files

- `train_lora.py` - Main training script
- `inference_with_lora.py` - Inference script using trained LoRA weights
- `configs/train/lora_finetune.yaml` - Configuration file for training
- `prepare_data.py` - Helper script to prepare training data

## Requirements

Install additional dependencies for training:

```bash
pip install accelerate
pip install tensorboard
```

All other dependencies should already be installed from the main MIDI-3D requirements.

## Data Preparation

### Expected Data Format

Your training data should be organized as follows:

```
data/training_data/
├── scene_0001/
│   ├── rgb.png          # Instance RGB image (512x512)
│   ├── seg.png          # Segmentation mask (512x512, grayscale)
│   ├── scene_rgb.png    # Full scene RGB image (512x512)
│   └── surface.npy      # Ground truth surface points [N, 3] or [N, 4]
├── scene_0002/
│   ├── rgb.png
│   ├── seg.png
│   ├── scene_rgb.png
│   └── surface.npy
└── ...
```

### Surface Points Format

The `surface.npy` file should contain a NumPy array of shape `[N, 3]` or `[N, 4]`:
- First 3 columns: XYZ coordinates of surface points
- Optional 4th column: Normal vectors (if available)
- Recommended: 20,480 points per object
- Points should be normalized to approximately [-1, 1] range

### Preparing Your Data

If you have 3D meshes, use the provided script:

```bash
python prepare_data.py --input_dir data/raw_meshes \
                       --output_dir data/training_data \
                       --num_samples 20480
```

## Training

### Quick Start

1. **Edit the configuration file** (`configs/train/lora_finetune.yaml`):
   - Update `data_dir` to point to your training data
   - Adjust batch size based on your GPU memory
   - Modify LoRA parameters if needed

2. **Start training**:

```bash
# Single GPU
python train_lora.py --config configs/train/lora_finetune.yaml

# Or with command line arguments
python train_lora.py \
    --pretrained_model pretrained_weights/MIDI-3D \
    --data_dir data/training_data \
    --output_dir outputs/lora_finetune \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 16
```

3. **Multi-GPU training**:

```bash
accelerate config  # Run once to configure
accelerate launch train_lora.py --config configs/train/lora_finetune.yaml
```

### Key Training Parameters

#### LoRA Configuration
- `--lora_rank` (default: 16): Rank of LoRA matrices. Higher = more capacity but slower
  - Use 8 for quick experiments
  - Use 16-32 for production
  - Use 64+ for complex adaptations
- `--lora_alpha` (default: 16): Scaling factor. Usually set equal to rank
- `--lora_dropout` (default: 0.1): Dropout rate for LoRA layers

#### Training Settings
- `--learning_rate` (default: 1e-4): Learning rate for LoRA parameters
  - Start with 1e-4
  - Increase to 5e-4 if training is too slow
  - Decrease to 5e-5 if unstable
- `--batch_size` (default: 4): Batch size per GPU
  - Reduce if out of memory
  - Increase if GPU is underutilized
- `--gradient_accumulation_steps` (default: 1): Simulate larger batch sizes
- `--gradient_checkpointing`: Enable to save memory (slower but uses less VRAM)

#### Multi-Instance Attention
- `--enable_multi_instance_attention`: Enable custom attention for multi-object scenes
- `--mi_attention_blocks`: Which transformer blocks to apply MI attention
  - Default: `["blocks.8", "blocks.9", "blocks.10", "blocks.11", "blocks.12"]`
  - These are the later layers for high-level reasoning
  - Add earlier blocks (0-7) for low-level detail control

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/lora_finetune
```

## Inference with Trained LoRA

After training, use your LoRA weights for inference:

```bash
python inference_with_lora.py \
    --lora_path outputs/lora_finetune/lora-final/lora_weights.pth \
    --rgb_image assets/example_data/Cartoon-Style/00_rgb.png \
    --seg_image assets/example_data/Cartoon-Style/00_seg.png \
    --output output.glb
```

### Advanced Inference Options

```bash
python inference_with_lora.py \
    --lora_path outputs/lora_finetune/lora-epoch-50/lora_weights.pth \
    --rgb_image my_image.png \
    --seg_image my_mask.png \
    --output my_scene.glb \
    --num_inference_steps 100 \  # More steps = better quality
    --guidance_scale 7.0 \         # Higher = stronger conditioning
    --seed 42
```

## Tips and Best Practices

### 1. **Start Small**
- Begin with a small dataset (50-100 scenes)
- Use low rank (8-16) for initial experiments
- Train for fewer epochs (10-20) to validate setup

### 2. **GPU Memory Management**
Based on GPU VRAM:
- **8GB**: batch_size=1, gradient_checkpointing=True, lora_rank=8
- **12GB**: batch_size=2, gradient_checkpointing=True, lora_rank=16
- **16GB**: batch_size=4, gradient_checkpointing=False, lora_rank=16
- **24GB+**: batch_size=8, gradient_checkpointing=False, lora_rank=32

### 3. **Hyperparameter Tuning**
- If underfitting (poor quality):
  - Increase `lora_rank` to 32 or 64
  - Increase `learning_rate` to 2e-4 or 5e-4
  - Train for more epochs
- If overfitting:
  - Decrease `lora_rank` to 8
  - Increase `lora_dropout` to 0.2
  - Add more training data

### 4. **Domain Adaptation**
For specific domains (e.g., furniture, cars, characters):
- Collect 200+ scenes from your domain
- Use higher LoRA rank (32-64)
- Train for 100-200 epochs
- Monitor validation loss carefully

### 5. **Multi-Instance Scenes**
For better multi-object generation:
- Ensure your training data has varied object counts
- Keep `enable_multi_instance_attention=True`
- Consider adding earlier transformer blocks to `mi_attention_blocks`

### 6. **Checkpointing Strategy**
The script saves:
- Full training state every `checkpointing_steps` (default: 500)
- LoRA weights every `save_every_n_epochs` (default: 10)
- Use intermediate checkpoints to find best model

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Reduce `lora_rank`
- Use mixed precision: `--mixed_precision fp16`

### Training is Slow
- Disable `gradient_checkpointing`
- Increase `batch_size`
- Use multiple GPUs with `accelerate launch`

### Poor Generation Quality
- Check your data quality and format
- Increase training epochs
- Adjust `guidance_scale` during inference (try 5.0-10.0)
- Increase `num_inference_steps` (try 100-200)

### NaN Loss
- Decrease learning rate to 5e-5 or 1e-5
- Enable gradient clipping (already enabled by default)
- Check your training data for corrupted samples

## Example Training Session

```bash
# 1. Prepare environment
conda activate midi
cd MIDI-3D

# 2. Prepare data (if needed)
python prepare_data.py --input_dir data/my_meshes --output_dir data/training_data

# 3. Edit config
nano configs/train/lora_finetune.yaml
# Update data_dir: "data/training_data"

# 4. Start training
python train_lora.py \
    --data_dir data/training_data \
    --output_dir outputs/my_lora \
    --batch_size 4 \
    --num_epochs 50 \
    --lora_rank 16

# 5. Monitor training
tensorboard --logdir outputs/my_lora

# 6. Test inference
python inference_with_lora.py \
    --lora_path outputs/my_lora/lora-final/lora_weights.pth \
    --rgb_image test_image.png \
    --seg_image test_mask.png \
    --output result.glb
```

## Citation

If you use this training script, please cite the original MIDI-3D paper:

```bibtex
@article{midi3d2024,
  title={MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This training script is released under the same license as MIDI-3D.
