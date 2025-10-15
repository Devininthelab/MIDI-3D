from data_preprocessing.simulation_utils import simulate_scenes
# # Simple example: simulate a mesh and output a video
# mesh_path = 'tmp/mesh_0.glb'
# output_video = 'tmp/simulation_output.mp4'

# simulate_and_visualize(
#     mesh_path=mesh_path,
#     output_video=output_video,
#     up_dir='y',
#     duration=10.0,
#     fps=30
# )

mesh_path = "tmp/midi3d_43a9bc4f-6f93-48e7-807e-fbb7b3381a18.glb"
angles = simulate_scenes([mesh_path], up_dir='y')
print(f"Stable angles: {angles}")
