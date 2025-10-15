import numpy as np
import trimesh
import mujoco
import mujoco.viewer
import cv2
from typing import Union, List
from data_preprocessing.simulation_utils import get_mujoco_model_and_data


def simulate_and_visualize(mesh_path, output_video='simulation.mp4', up_dir='y', duration=10.0, fps=30):
    """
    Simulate a mesh falling and settling, then output a video of the simulation.
    
    Args:
        mesh_path: Path to the mesh file (.glb, .obj, etc.)
        output_video: Path for the output video file
        up_dir: Up direction of the mesh ('x', 'y', or 'z')
        duration: Simulation duration in seconds
        fps: Frames per second for the output video
    """
    
    # Load mesh
    mesh = trimesh.util.concatenate(trimesh.load(mesh_path))
    
    if isinstance(mesh, trimesh.points.PointCloud):
        print(f"Mesh is a PointCloud, cannot simulate.")
        return None
    
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    
    # Transform vertices based on up direction
    vertices = np.array(mesh.vertices)
    if up_dir == "y":
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
    elif up_dir == "x":
        vertices[:, [0, 2]] = vertices[:, [2, 0]]
    
    vertices[:, 2] -= np.min(vertices[:, 2])  # align bottom to z=0
    faces = np.array(mesh.faces).astype(np.int32)
    
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        print(f"Mesh has no vertices or faces, cannot simulate.")
        return None
    
    # Create MuJoCo model and data
    model, data = get_mujoco_model_and_data(vertices, faces)
    model.opt.timestep = 0.01
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Setup camera (use default camera or specify position)
    renderer.update_scene(data)
    
    frames = []
    frame_interval = 1.0 / fps
    next_frame_time = 0.0
    
    print(f"Simulating for {duration} seconds at {fps} fps...")
    
    # Run simulation and capture frames
    while data.time < duration:
        mujoco.mj_step(model, data)
        
        # Capture frame at specified fps
        if data.time >= next_frame_time:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels.copy())
            next_frame_time += frame_interval
    
    # Save video
    print(f"Saving video to {output_video}...")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"Video saved successfully! Total frames: {len(frames)}")
    
    return output_video


if __name__ == "__main__":
    # Example usage
    mesh_path = "tmp/mesh_0.glb"
    output_video = "tmp/simulation_output.mp4"
    
    simulate_and_visualize(
        mesh_path=mesh_path,
        output_video=output_video,
        up_dir='y',
        duration=10.0,
        fps=30
    )
