"""
Inference script to use the finetuned LoRA weights with MIDI-3D

Usage:
    python inference_with_lora.py --lora_path outputs/lora_finetune/lora-final/lora_weights.pth \
                                   --rgb_image assets/example_data/Cartoon-Style/00_rgb.png \
                                   --seg_image assets/example_data/Cartoon-Style/00_seg.png \
                                   --output output.glb
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from skimage import measure

from midi.pipelines.pipeline_midi import MIDIPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with LoRA finetuned MIDI-3D")
    
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="pretrained_weights/MIDI-3D",
        help="Path to pretrained MIDI-3D model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights (.pth file)"
    )
    parser.add_argument(
        "--rgb_image",
        type=str,
        required=True,
        help="Path to RGB input image"
    )
    parser.add_argument(
        "--seg_image",
        type=str,
        required=True,
        help="Path to segmentation mask image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.glb",
        help="Output path for 3D model"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (must match training config)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (must match training config)"
    )
    parser.add_argument(
        "--mi_attention_blocks",
        type=str,
        nargs="+",
        default=["blocks.8", "blocks.9", "blocks.10", "blocks.11", "blocks.12"],
        help="Transformer blocks with multi-instance attention"
    )
    
    return parser.parse_args()


def load_lora_weights(pipeline: MIDIPipeline, lora_path: str, lora_config: dict):
    """Load LoRA weights into the pipeline"""
    # Initialize the adapter structure first
    pipeline.init_custom_adapter(
        set_self_attn_module_names=lora_config.get("mi_attention_blocks"),
        transformer_lora_config={
            "r": lora_config["lora_rank"],
            "lora_alpha": lora_config["lora_alpha"],
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        },
    )
    
    # Load the LoRA weights
    lora_state_dict = torch.load(lora_path, map_location="cpu")
    
    # Load into transformer
    pipeline.transformer.load_state_dict(lora_state_dict, strict=False)
    
    print(f"Successfully loaded LoRA weights from {lora_path}")
    return pipeline


def preprocess_images(rgb_path: str, seg_path: str):
    """Load and preprocess input images"""
    rgb_image = Image.open(rgb_path).convert("RGB")
    seg_image = Image.open(seg_path).convert("L")
    
    # Calculate number of instances
    seg_array = np.array(seg_image)
    unique_labels = np.unique(seg_array)
    num_instances = len(unique_labels[unique_labels > 0])
    
    print(f"Detected {num_instances} instances in the scene")
    
    return rgb_image, seg_image, num_instances


def generate_scene(
    pipeline: MIDIPipeline,
    rgb_image: Image.Image,
    seg_image: Image.Image,
    num_instances: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    seed: int = 42,
    device: str = "cuda"
):
    """Generate 3D scene from images"""
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Run the pipeline
    output = pipeline(
        image=rgb_image,
        mask=seg_image,
        image_scene=rgb_image,  # Use same image as scene
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        attention_kwargs={
            "num_instances": torch.tensor([num_instances]),
            "num_instances_per_batch": torch.tensor([num_instances]),
        },
        decode_progressive=True,
        return_dict=True,
    )
    
    return output


def save_mesh(output, output_path: str):
    """Convert output to mesh and save"""
    meshes = []
    
    for sdf, grid_size, bbox_size, bbox_min, bbox_max in zip(
        output.samples,
        output.grid_sizes,
        output.bbox_sizes,
        output.bbox_mins,
        output.bbox_maxs
    ):
        # Reshape SDF to grid
        grid_sdf = sdf.view(grid_size).float().cpu().numpy()
        
        # Marching cubes
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_sdf, level=0, method="lewiner"
        )
        
        # Scale vertices to world space
        vertices = vertices / grid_size * bbox_size + bbox_min
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=faces,
            process=False
        )
        meshes.append(mesh)
    
    # Combine all meshes
    if len(meshes) > 1:
        combined_mesh = trimesh.util.concatenate(meshes)
    else:
        combined_mesh = meshes[0]
    
    # Save
    combined_mesh.export(output_path)
    print(f"Saved 3D scene to {output_path}")
    
    # Print statistics
    print(f"Total vertices: {len(combined_mesh.vertices)}")
    print(f"Total faces: {len(combined_mesh.faces)}")
    print(f"Number of objects: {len(meshes)}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained pipeline
    print(f"Loading pretrained model from {args.pretrained_model}")
    pipeline = MIDIPipeline.from_pretrained(args.pretrained_model)
    pipeline = pipeline.to(device, torch.float16 if device.type == "cuda" else torch.float32)
    
    # Load LoRA weights
    lora_config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "mi_attention_blocks": args.mi_attention_blocks,
    }
    pipeline = load_lora_weights(pipeline, args.lora_path, lora_config)
    
    # Preprocess images
    print(f"Loading images...")
    rgb_image, seg_image, num_instances = preprocess_images(args.rgb_image, args.seg_image)
    
    # Generate 3D scene
    print(f"Generating 3D scene with {args.num_inference_steps} steps...")
    output = generate_scene(
        pipeline,
        rgb_image,
        seg_image,
        num_instances,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device
    )
    
    # Save output
    print(f"Saving output...")
    save_mesh(output, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
