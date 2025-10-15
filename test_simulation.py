from data_preprocessing.simulation_utils import simulate_models
import trimesh

mesh_path = 'tmp/midi3d_a7095494-f06f-43ae-bc3b-1202d8ec5bdd.glb'

mesh_obj = trimesh.load(mesh_path)
print(mesh_obj)
# If it's a Scene, dump to list of meshes
if isinstance(mesh_obj, trimesh.Scene):
    meshes = mesh_obj.dump()
    print(f"Number of meshes: {len(meshes)}")
    for i, mesh in enumerate(meshes):
        print(f"Mesh {i}: {mesh}")


angles = simulate_models([mesh_path], up_dir="y", num_workers=1)
print(angles)