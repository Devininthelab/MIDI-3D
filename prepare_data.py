"""
Data preparation script for MIDI-3D LoRA finetuning

This script helps convert your 3D meshes into the format required for training.
It samples surface points from meshes and organizes them with corresponding images.

Usage:
    python prepare_data.py --input_dir data/raw_meshes \
                           --output_dir data/training_data \
                           --num_samples 20480
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data for MIDI-3D LoRA finetuning")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw mesh files and images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed training data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20480,
        help="Number of surface points to sample from each mesh"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize mesh vertices to [-1, 1] range"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize images to this size"
    )
    
    return parser.parse_args()


def normalize_mesh(mesh):
    """Normalize mesh vertices to approximately [-1, 1] range"""
    vertices = mesh.vertices
    
    # Center the mesh
    center = vertices.mean(axis=0)
    vertices = vertices - center
    
    # Scale to [-1, 1]
    max_dist = np.abs(vertices).max()
    if max_dist > 0:
        vertices = vertices / max_dist * 0.95  # Scale to 95% to leave some margin
    
    mesh.vertices = vertices
    return mesh


def sample_surface_points(mesh, num_samples=20480, include_normals=True):
    """Sample points uniformly from mesh surface"""
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    
    if include_normals:
        # Get normals at sampled points
        normals = mesh.face_normals[face_indices]
        surface_data = np.concatenate([points, normals], axis=1)  # [N, 6]
    else:
        surface_data = points  # [N, 3]
    
    return surface_data


def process_mesh_file(mesh_path, num_samples=20480, normalize=True):
    """Load and process a single mesh file"""
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Handle multiple meshes (if loaded from complex file)
        if isinstance(mesh, trimesh.Scene):
            # Combine all meshes in the scene
            mesh = trimesh.util.concatenate([
                geom for geom in mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh)
            ])
        
        # Normalize if requested
        if normalize:
            mesh = normalize_mesh(mesh)
        
        # Sample surface points
        surface_points = sample_surface_points(mesh, num_samples, include_normals=False)
        
        return surface_points, mesh
        
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None, None


def resize_image(image_path, size=512):
    """Load and resize image"""
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB' and 'seg' not in str(image_path) and 'mask' not in str(image_path):
        img = img.convert('RGB')
    elif 'seg' in str(image_path) or 'mask' in str(image_path):
        img = img.convert('L')
    
    # Resize
    if img.size != (size, size):
        resample = Image.NEAREST if img.mode == 'L' else Image.LANCZOS
        img = img.resize((size, size), resample)
    
    return img


def process_scene(input_scene_dir, output_scene_dir, num_samples, image_size, normalize):
    """
    Process a single scene directory.
    
    Expected input structure:
        input_scene_dir/
            mesh.obj (or .glb, .ply, etc.)
            rgb.png
            seg.png (optional, will copy if exists)
            scene_rgb.png (optional, will use rgb.png if not exists)
    """
    os.makedirs(output_scene_dir, exist_ok=True)
    
    # Find mesh file
    mesh_extensions = ['.obj', '.glb', '.ply', '.stl', '.off']
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(list(input_scene_dir.glob(f'*{ext}')))
    
    if len(mesh_files) == 0:
        print(f"No mesh file found in {input_scene_dir}")
        return False
    
    mesh_path = mesh_files[0]
    
    # Process mesh
    surface_points, mesh = process_mesh_file(mesh_path, num_samples, normalize)
    if surface_points is None:
        return False
    
    # Save surface points
    np.save(output_scene_dir / 'surface.npy', surface_points)
    
    # Process images
    # RGB image
    rgb_path = input_scene_dir / 'rgb.png'
    if not rgb_path.exists():
        # Try other common names
        for name in ['image.png', 'render.png', 'color.png']:
            if (input_scene_dir / name).exists():
                rgb_path = input_scene_dir / name
                break
    
    if rgb_path.exists():
        rgb_img = resize_image(rgb_path, image_size)
        rgb_img.save(output_scene_dir / 'rgb.png')
    else:
        print(f"Warning: No RGB image found in {input_scene_dir}")
        return False
    
    # Segmentation image
    seg_path = input_scene_dir / 'seg.png'
    if not seg_path.exists():
        # Try other names
        for name in ['mask.png', 'segmentation.png']:
            if (input_scene_dir / name).exists():
                seg_path = input_scene_dir / name
                break
    
    if seg_path.exists():
        seg_img = resize_image(seg_path, image_size)
        seg_img.save(output_scene_dir / 'seg.png')
    else:
        print(f"Warning: No segmentation found in {input_scene_dir}, creating dummy mask")
        # Create a simple mask (entire image)
        seg_img = Image.new('L', (image_size, image_size), 255)
        seg_img.save(output_scene_dir / 'seg.png')
    
    # Scene RGB (use rgb if not exists)
    scene_rgb_path = input_scene_dir / 'scene_rgb.png'
    if scene_rgb_path.exists():
        scene_rgb_img = resize_image(scene_rgb_path, image_size)
        scene_rgb_img.save(output_scene_dir / 'scene_rgb.png')
    else:
        # Use same as rgb
        shutil.copy(output_scene_dir / 'rgb.png', output_scene_dir / 'scene_rgb.png')
    
    return True


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all scene directories
    scene_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if len(scene_dirs) == 0:
        print(f"No subdirectories found in {input_dir}")
        print("Expected structure: input_dir/scene_001/, input_dir/scene_002/, etc.")
        return
    
    print(f"Found {len(scene_dirs)} scenes to process")
    
    # Process each scene
    successful = 0
    failed = 0
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = scene_dir.name
        output_scene_dir = output_dir / scene_name
        
        success = process_scene(
            scene_dir,
            output_scene_dir,
            args.num_samples,
            args.image_size,
            args.normalize
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            # Remove incomplete output directory
            if output_scene_dir.exists():
                shutil.rmtree(output_scene_dir)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} scenes")
    print(f"Failed: {failed} scenes")
    print(f"Output saved to: {output_dir}")
    
    # Create a summary file
    summary = {
        "total_scenes": successful,
        "num_surface_points": args.num_samples,
        "image_size": args.image_size,
        "normalized": args.normalize,
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nYou can now train with: --data_dir {output_dir}")


if __name__ == "__main__":
    main()
