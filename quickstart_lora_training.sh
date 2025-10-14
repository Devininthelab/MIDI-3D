#!/bin/bash
# Quick start script for MIDI-3D LoRA finetuning
# This script will guide you through the setup and training process

set -e  # Exit on error

echo "=================================="
echo "MIDI-3D LoRA Finetuning Setup"
echo "=================================="
echo ""

# Check if conda environment exists
if ! conda info --envs | grep -q "midi"; then
    echo "Creating conda environment 'midi'..."
    conda create -n midi python=3.10 -y
    echo "✓ Conda environment created"
else
    echo "✓ Conda environment 'midi' already exists"
fi

# Activate environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate midi

# Install dependencies
echo ""
echo "Installing dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

if ! python -c "import accelerate" 2>/dev/null; then
    echo "Installing training dependencies..."
    pip install accelerate tensorboard
fi

if [ ! -d "pretrained_weights/MIDI-3D" ]; then
    echo "Installing base dependencies..."
    pip install -r requirements.txt
fi

echo "✓ Dependencies installed"

# Download pretrained model if not exists
echo ""
if [ ! -d "pretrained_weights/MIDI-3D" ]; then
    echo "Downloading pretrained MIDI-3D model..."
    echo "This will be done automatically on first run of training script"
else
    echo "✓ Pretrained model already downloaded"
fi

# Check for data
echo ""
echo "=================================="
echo "Data Setup"
echo "=================================="
echo ""
echo "Please ensure your training data is prepared in the following structure:"
echo ""
echo "data/training_data/"
echo "├── scene_0001/"
echo "│   ├── rgb.png"
echo "│   ├── seg.png"
echo "│   ├── scene_rgb.png"
echo "│   └── surface.npy"
echo "├── scene_0002/"
echo "│   └── ..."
echo ""
read -p "Do you have your data ready? (y/n): " data_ready

if [ "$data_ready" != "y" ]; then
    echo ""
    echo "To prepare your data, use:"
    echo "  python prepare_data.py --input_dir YOUR_DATA --output_dir data/training_data"
    echo ""
    echo "See TRAINING_README.md for detailed instructions"
    exit 0
fi

# Get data directory
read -p "Enter path to your training data [data/training_data]: " data_dir
data_dir=${data_dir:-data/training_data}

if [ ! -d "$data_dir" ]; then
    echo "Error: Directory $data_dir does not exist"
    exit 1
fi

# Count scenes
num_scenes=$(find "$data_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "✓ Found $num_scenes scenes in $data_dir"

# Configure training
echo ""
echo "=================================="
echo "Training Configuration"
echo "=================================="
echo ""

read -p "Output directory [outputs/lora_finetune]: " output_dir
output_dir=${output_dir:-outputs/lora_finetune}

read -p "Batch size [4]: " batch_size
batch_size=${batch_size:-4}

read -p "Number of epochs [100]: " num_epochs
num_epochs=${num_epochs:-100}

read -p "Learning rate [1e-4]: " learning_rate
learning_rate=${learning_rate:-1e-4}

read -p "LoRA rank [16]: " lora_rank
lora_rank=${lora_rank:-16}

read -p "Enable gradient checkpointing for memory saving? (y/n) [n]: " grad_ckpt
if [ "$grad_ckpt" == "y" ]; then
    grad_ckpt_flag="--gradient_checkpointing"
else
    grad_ckpt_flag=""
fi

# Ask about multi-GPU
echo ""
read -p "Use multi-GPU training with accelerate? (y/n) [n]: " use_accelerate

if [ "$use_accelerate" == "y" ]; then
    if ! accelerate env 2>/dev/null | grep -q "Num processes"; then
        echo ""
        echo "Running accelerate config..."
        echo "Please answer the questions to configure multi-GPU training"
        accelerate config
    fi
fi

# Create output directory
mkdir -p "$output_dir"

# Summary
echo ""
echo "=================================="
echo "Training Summary"
echo "=================================="
echo "Data directory: $data_dir"
echo "Output directory: $output_dir"
echo "Number of scenes: $num_scenes"
echo "Batch size: $batch_size"
echo "Epochs: $num_epochs"
echo "Learning rate: $learning_rate"
echo "LoRA rank: $lora_rank"
echo "Gradient checkpointing: ${grad_ckpt:-disabled}"
echo "Multi-GPU: ${use_accelerate:-no}"
echo "=================================="
echo ""

read -p "Start training? (y/n): " start_training

if [ "$start_training" != "y" ]; then
    echo "Training cancelled"
    exit 0
fi

# Build command
train_cmd="python train_lora.py \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --learning_rate $learning_rate \
    --lora_rank $lora_rank \
    $grad_ckpt_flag"

if [ "$use_accelerate" == "y" ]; then
    train_cmd="accelerate launch $train_cmd"
fi

echo ""
echo "Starting training with command:"
echo "$train_cmd"
echo ""

# Save training command for reference
echo "$train_cmd" > "$output_dir/training_command.sh"
chmod +x "$output_dir/training_command.sh"

# Start training
eval $train_cmd

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Your LoRA weights are saved in: $output_dir"
echo ""
echo "To use them for inference:"
echo "  python inference_with_lora.py \\"
echo "    --lora_path $output_dir/lora-final/lora_weights.pth \\"
echo "    --rgb_image YOUR_IMAGE.png \\"
echo "    --seg_image YOUR_MASK.png \\"
echo "    --output output.glb"
echo ""
echo "To monitor training (in another terminal):"
echo "  tensorboard --logdir $output_dir"
echo ""
