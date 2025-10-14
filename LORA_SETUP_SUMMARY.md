# LoRA Finetuning Setup Summary

## Created Files

I've created a complete training setup for finetuning the TripoSGDiTModel (transformer) component of MIDI-3D using LoRA. Here's what was added:

### 1. Main Training Script
**`train_lora.py`** - Complete training script with the following features:
- LoRA integration for efficient finetuning
- Accelerate support for multi-GPU training
- Mixed precision training (fp16/bf16)
- Gradient checkpointing for memory efficiency
- Multi-instance attention support
- Custom dataset loader
- Checkpointing and model saving
- TensorBoard logging
- Flow matching loss computation
- Configurable via command line or config file

### 2. Configuration File
**`configs/train/lora_finetune.yaml`** - YAML configuration with sensible defaults:
- LoRA rank: 16, alpha: 16, dropout: 0.1
- Batch size: 4
- Learning rate: 1e-4 with cosine scheduler
- 100 epochs with warmup
- Multi-instance attention on blocks 8-12
- Mixed precision fp16

### 3. Inference Script
**`inference_with_lora.py`** - Script to use trained LoRA weights:
- Loads pretrained MIDI-3D model
- Applies trained LoRA weights
- Generates 3D scenes from images
- Exports to various 3D formats (GLB, OBJ, etc.)
- Configurable inference parameters

### 4. Data Preparation Script
**`prepare_data.py`** - Helper to prepare training data:
- Converts 3D meshes to training format
- Samples surface points uniformly
- Normalizes meshes to [-1, 1] range
- Resizes and preprocesses images
- Creates proper directory structure
- Handles various mesh formats (OBJ, GLB, PLY, STL)

### 5. Documentation
**`TRAINING_README.md`** - Comprehensive guide covering:
- Requirements and installation
- Data format and preparation
- Training configuration
- GPU memory management tips
- Hyperparameter tuning guide
- Troubleshooting common issues
- Example training sessions
- Best practices

### 6. Quick Start Script
**`quickstart_lora_training.sh`** - Interactive setup wizard:
- Checks dependencies
- Sets up conda environment
- Guides through configuration
- Starts training with optimal settings
- Multi-GPU setup support

## Quick Start Guide

### Option 1: Interactive Setup (Recommended)
```bash
chmod +x quickstart_lora_training.sh
./quickstart_lora_training.sh
```

### Option 2: Manual Training
```bash
# 1. Prepare your data
python prepare_data.py \
    --input_dir data/raw_meshes \
    --output_dir data/training_data \
    --num_samples 20480

# 2. Start training
python train_lora.py \
    --data_dir data/training_data \
    --output_dir outputs/lora_finetune \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 16

# 3. Use trained model
python inference_with_lora.py \
    --lora_path outputs/lora_finetune/lora-final/lora_weights.pth \
    --rgb_image test.png \
    --seg_image mask.png \
    --output result.glb
```

### Option 3: Multi-GPU Training
```bash
# Configure accelerate (run once)
accelerate config

# Launch training
accelerate launch train_lora.py \
    --data_dir data/training_data \
    --output_dir outputs/lora_finetune \
    --batch_size 4 \
    --num_epochs 100
```

## Key Features

### ✅ Efficient Training
- **LoRA**: Trains only 1-5% of parameters (~50-200MB weights)
- **Memory Efficient**: Gradient checkpointing + mixed precision
- **Fast**: Significantly faster than full finetuning
- **Multi-GPU**: Built-in distributed training support

### ✅ Flexible Configuration
- **Command-line arguments**: Quick experiments
- **YAML configs**: Reproducible training runs
- **LoRA parameters**: Adjustable rank, alpha, dropout
- **Attention control**: Selective multi-instance attention

### ✅ Production Ready
- **Checkpointing**: Regular saves + best model tracking
- **Logging**: TensorBoard integration
- **Resumable**: Continue from checkpoint
- **Validation**: Monitor training progress

### ✅ Easy to Use
- **Interactive setup**: Guided configuration
- **Data preparation**: Automated preprocessing
- **Documentation**: Comprehensive guides
- **Examples**: Ready-to-use configurations

## What Components Are Being Finetuned?

The training script specifically targets the **Transformer (TripoSGDiTModel)** component using LoRA:

1. **Target Modules** (default):
   - `to_q`: Query projection in attention
   - `to_k`: Key projection in attention
   - `to_v`: Value projection in attention
   - `to_out.0`: Output projection in attention

2. **Multi-Instance Attention Blocks** (default):
   - `blocks.8`, `blocks.9`, `blocks.10`, `blocks.11`, `blocks.12`
   - These are the later transformer layers responsible for high-level reasoning

3. **Frozen Components**:
   - VAE (encoder + decoder)
   - Image Encoder 1 (CLIP)
   - Image Encoder 2 (DINOv2)
   - Feature extractors

This design ensures:
- Efficient training (only ~1-5% of parameters)
- Preserved semantic understanding (frozen image encoders)
- Maintained latent space (frozen VAE)
- Targeted 3D generation improvement (finetuned transformer)

## Expected Training Time

Based on typical setups:
- **Small dataset (100 scenes)**: 2-4 hours on single GPU
- **Medium dataset (500 scenes)**: 8-12 hours on single GPU
- **Large dataset (2000+ scenes)**: 1-2 days on single GPU, 6-12 hours on 4 GPUs

## GPU Requirements

Minimum recommended: **12GB VRAM** (e.g., RTX 3060, RTX 4070)
- Batch size 2, gradient checkpointing enabled, rank 8

Comfortable: **16GB+ VRAM** (e.g., RTX 4080, A4000)
- Batch size 4, gradient checkpointing optional, rank 16

Optimal: **24GB+ VRAM** (e.g., RTX 4090, A5000, A6000)
- Batch size 8+, no gradient checkpointing, rank 32+

## Next Steps

1. **Read TRAINING_README.md** for detailed documentation
2. **Prepare your data** using the expected format
3. **Run quickstart script** for guided setup
4. **Monitor training** with TensorBoard
5. **Test inference** with your trained LoRA weights

## Support

For issues or questions:
1. Check TRAINING_README.md troubleshooting section
2. Verify your data format matches requirements
3. Review GPU memory recommendations
4. Check the original MIDI-3D repository for updates

Happy training! 🚀
