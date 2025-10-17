from glob import glob
import os

import argparse
from PIL import Image
import torch
from tqdm import tqdm

from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.inference_midi import run_midi_and_return_latents

# python -m data_preprocessing.generate_synthetic_data --num_samples 4 --save_extra --image_paths "./category/*/**.png"

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0) # Job id for multiple jobs
parser.add_argument("--num_jobs", type=int, default=1) # Total number of jobs
parser.add_argument("--num_samples", type=int, default=1) # Total number of samples
parser.add_argument("--save_extra", action="store_true") # Whether to save extra outputs
parser.add_argument("--output_dir", type=str, default="./synthetic_data") # Output directory
parser.add_argument("--image_paths", type=str, required=True) # Input image paths (can be a glob pattern)

args = parser.parse_args()


# Load a pipeline from the pretrained weights dir
## MIDI-3D
local_dir = "pretrained_weights/MIDI-3D"
DEVICE = "cuda"
DTYPE = torch.float16
pipe: MIDIPipeline = MIDIPipeline.from_pretrained(local_dir).to(DEVICE, DTYPE)
pipe.init_custom_adapter(
    set_self_attn_module_names=[
        "blocks.8",
        "blocks.9",
        "blocks.10",
        "blocks.11",
        "blocks.12",
    ]
)

sample_batch_size, num_batches = 1, args.num_samples

if isinstance(args.image_paths, str):
    images = sorted(glob(args.image_paths))
else:
    images = sorted(args.image_paths)

rgb_images = sorted(i for i in images if 'rgb' in i)
seg_images = sorted(i for i in images if 'seg' in i)
assert len(rgb_images) == len(seg_images), f"Number of rgb images ({len(rgb_images)}) and seg images ({len(seg_images)}) must be the same"
images = list(zip(rgb_images, seg_images))

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"################ Total scenes to generate: {len(images)} ################")


for image_path in tqdm(images[args.job_id::args.num_jobs]):
    rgb_image_path, seg_image_path = image_path
    
    category, prompt, image_name = rgb_image_path.split("/")[-3:]

    save_root = os.path.join(output_dir, f"{category}/{prompt}")
    os.makedirs(save_root, exist_ok=True)
    generate_3d = True
    if len(glob(os.path.join(save_root, f"{image_name.replace('_rgb.jpg', '_*.glb').replace('_rgb.png', '_*.glb')}"))) >= args.num_samples:
        generate_3d = False
    
    if generate_3d:
        for bid in range(num_batches):
            # Check if all samples in this batch already exist
            skip_batch = True
            for i in range(sample_batch_size):
                eid = bid * sample_batch_size + i
                glb_path = os.path.join(save_root, f"{image_name.replace('_rgb.jpg', f'_{eid:03d}.glb').replace('_rgb.png', f'_{eid:03d}.glb')}")
                sparse_sample_path = os.path.join(save_root, f"{image_name.replace('_rgb.jpg', f'_sparse_sample_{eid:03d}.pt').replace('_rgb.png', f'_sparse_sample_{eid:03d}.pt')}") #latents before decoding
                if not os.path.exists(glb_path) or not os.path.exists(sparse_sample_path):
                    skip_batch = False
                    break
            if skip_batch:
                continue

            success = False
            # Run the pipeline
            # SHOULD OPTIMIZE TO RUN SCENE IN BATCH: CONSIDER HEREREEEEEEE
            try:
                scene, latents, cond = run_midi_and_return_latents(
                    pipe,
                    rgb_image_path,
                    seg_image_path,
                    seed=bid,
                    num_inference_steps=50,
                    guidance_scale=7.0,
                    do_image_padding=True,
                )
                success = True
            except Exception as e:
                print(e)
                pass
            
            if not success:
                print(f"Failed to generate 3D models for {image_path}")
                continue

            if args.save_extra:
                if bid == 0:
                    # Cache the image cond
                    data_cpu = {k: v.cpu() for k, v in cond.items()}
                    image_cond_save_path = os.path.join(save_root, f"{image_name.replace('_rgb.jpg', '_cond.pt').replace('_rgb.png', '_cond.pt')}")
                    torch.save(data_cpu, image_cond_save_path)
            
                for i in range(sample_batch_size): # this loop is run 1 time only tho, not sure the purpose of it
                    eid = bid * sample_batch_size + i
                    sparse_sample = latents.cpu()
                    sparse_sample_save_path = os.path.join(save_root, f"{image_name.replace('_rgb.jpg', f'_sparse_sample_{eid:03d}.pt').replace('_rgb.png', f'_sparse_sample_{eid:03d}.pt')}")
                    torch.save(sparse_sample.cpu(), sparse_sample_save_path)
            # save the glb
            for i in range(sample_batch_size): # this loop is run 1 time only tho, not sure the purpose of it
                glb_path = os.path.join(save_root, f"{image_name.replace('_rgb.jpg', f'_{(bid*sample_batch_size+i):03d}.glb').replace('_rgb.png', f'_{(bid*sample_batch_size+i):03d}.glb')}")
                scene.export(glb_path)