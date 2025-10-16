import os
import random


import torch
import trimesh


from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
from scripts.image_to_textured_scene import (
    prepare_ig2mv_pipeline,
    prepare_texture_pipeline,
    run_i2tex,
)
from scripts.inference_midi import run_midi, preprocess_image, split_rgb_mask, run_midi_and_return_latents

# Test variables
seg_map_pil_path  = 'assets/example_data/BlendSwap/breakfast-room_seg.png'
img_pil_path     = 'assets/example_data/BlendSwap/breakfast-room_rgb.png'


# CONSTANTS
local_dir = "pretrained_weights/MIDI-3D"
DEVICE = "cuda"
DTYPE = torch.float16

# Init the pipe
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
# print(pipe.transformer)

# When run call method, set output_type = 'latent' to get the latent x0
do_image_padding = True
seed = 42
num_inference_steps = 50
guidance_scale = 7.0
with torch.no_grad():
    if do_image_padding:
        rgb_image, seg_image = preprocess_image(img_pil_path, seg_map_pil_path)
    instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image)
    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=pipe.device).manual_seed(seed)

    num_instances = len(instance_rgbs)
    print(f"Detected {num_instances} instances")
    outputs = pipe(
        image=instance_rgbs,
        mask=instance_masks,
        image_scene=scene_rgbs,
        attention_kwargs={"num_instances": num_instances},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        decode_progressive=True,
        output_type='latent',
        return_dict=True,
        **pipe_kwargs,
    )
    print(outputs.samples.shape)

    # ALso workS
    # scene, latents, cond = run_midi_and_return_latents(
    #     pipe,
    #     rgb_image=img_pil_path,
    #     seg_image=seg_map_pil_path,
    #     do_image_padding=do_image_padding,
    #     seed=seed,
    # )
    # print(type(scene))
    # print(cond)
    # print(torch.equal(latents, outputs.samples))  # True

# need to take the x0 before put to the vae

## MV-Adapter: NO NEED TO USE AS WE GENEREATE THE MESH, NOT THE TEXTURE
# ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
# texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)
