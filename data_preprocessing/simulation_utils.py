from contextlib import contextmanager
from functools import partial
import multiprocessing
import os
import signal 

from typing import List, Union

import mujoco
import numpy as np
import trimesh
from tqdm import tqdm


def quaternion_to_axis_angle(q):
    """Convert quaternion to axis-angle representation
    Quarterion are angular representation of rotation in 3D space
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    angle = 2 * np.arccos(w)
    sin_theta = np.sqrt(1 - w**2)  # sin(theta/2)
    
    zero_mask = sin_theta < 1e-6
    axis = np.zeros(q.shape[:-1] + (3,))
    
    axis = np.where(zero_mask[..., None], 
                    np.array([1., 0., 0.]), 
                    np.stack([x, y, z], axis=-1) / np.where(zero_mask[..., None], 1, sin_theta[..., None]))
    return axis, angle


def get_mujoco_model_and_data(
        v_pos: Union[np.ndarray, List[np.ndarray]],
        faces: Union[np.ndarray, List[np.ndarray]],
    ):
    """Create a MuJoCo model and data from a list of meshes (v_pos, faces)"""

    if not isinstance(v_pos, list):
        v_pos = [v_pos]
    if not isinstance(faces, list):
        faces = [faces]
    if len(v_pos) != len(faces):
        assert len(v_pos) == 1 or len(faces) == 1
        if len(v_pos) == 1:
            v_pos = v_pos * len(faces)
        else:
            faces = faces * len(v_pos)

    asset_xml, body_xml = "", ""
    for i, (vp, f) in enumerate(zip(v_pos, faces)):
        faces_str = "  ".join(f"{v1:d} {v2:d} {v3:d}" for v1, v2, v3 in f)
        vertices_str = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vp)
        asset_xml += f"""
            <mesh name="mesh_{i}" scale="1 1 1" vertex="{vertices_str}" face="{faces_str}"/>
        """
        body_xml += f"""
            <body name="mesh_body_{i}" pos="0 {i * 5} 0">
                <joint name="free_joint_{i}" type="free" />
                <geom name="mesh_geom_{i}" type="mesh" mesh="mesh_{i}"/>
            </body>
        """
    
    # Add a light source
    body_xml += """
            <light name="light1" pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    """

    model_xml = f"""
        <mujoco model="mesh_simulation">
            <asset>
                {asset_xml}
            </asset>

            <worldbody>
                <!-- Ground plane -->
                <geom name="ground_plane" type="plane" pos="0 0 0" size="0 0 1" />
                
                <!-- Mesh object with free joint --> 
                {body_xml}
            </worldbody>
        </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    return model, data

def get_sim_angles(v_pos: Union[np.ndarray, List[np.ndarray]], faces: Union[np.ndarray, List[np.ndarray]], timeout: float = 30.0):
    """Get stable angles of the meshes in simulation"""
    
    def simulate():
        # Convert vetices and faces to a format suitable for MujoCo
        model, data = get_mujoco_model_and_data(v_pos, faces)
        duration = 10.0 # seconds
        model.opt.timestep = 0.01 # timestep for simulation

        while data.time < duration:
            mujoco.mj_step(model, data)
        
        rotation = data.xquat[1:, :] # skip the first one which is the ground plane
        _, angles = quaternion_to_axis_angle(np.array(rotation))
        angles = np.rad2deg(angles)  # convert to degrees
        return angles
    
    try: 
        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Simulation timed out")
            
            # Set signal handler
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(int(timeout)) # must be integer

            try:
                yield
            finally:
                signal.alarm(0) # disable alarm
        with time_limit(timeout):
            return simulate()
    except TimeoutError:
        print("Simulation timed out")
        return None
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        return None
    
def simulate_single_model(mesh_path, up_dir):
    if not os.path.exists(mesh_path):
        return None
    
    try: 
        mesh = trimesh.util.concatenate(trimesh.load(mesh_path))

        if isinstance(mesh, trimesh.points.PointCloud):
            print(f"Mesh at {mesh_path} is a PointCloud, skipping.")
            return None

        mesh.update_faces(mesh.unique_faces())
        mesh.merge_vertices()
        
        
        vertices = np.array(mesh.vertices)
        if up_dir == "y":
            vertices[:, [1, 2]] = vertices[:, [2, 1]]
        elif up_dir == "x":
            vertices[:, [0, 2]] = vertices[:, [2, 0]]

        vertices[:, 2] -= np.min(vertices[:, 2]) # align bottom to z=0
        faces = np.array(mesh.faces).astype(np.int32)

        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            print(f"Mesh at {mesh_path} has no vertices or faces, skipping.")
            return None
        
        angle = get_sim_angles(vertices, faces)
        return angle
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return 
 
def simulate_models(mesh_paths, up_dir="y", num_workers=None):
    process_fn = partial(simulate_single_model, up_dir=up_dir)
    angles = []
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
    if num_workers == 1:
        for mesh_path in tqdm(mesh_paths):
            angle = process_fn(mesh_path)
            if angle is not None:
                angles.append(angle)
    else: 
        with multiprocessing.Pool(num_workers) as pool:
            angles = list(tqdm(pool.imap(process_fn, mesh_paths), total=len(mesh_paths)))
    valid_angles = [a for a in angles if a is not None]
    if len(valid_angles) < len(mesh_paths):
        return None
    return np.array(valid_angles)


##### FOR MIDI CODE ####
def simulate_single_model_mesh_obj(mesh_obj, up_dir):
    try: 
        mesh = trimesh.util.concatenate(mesh_obj)

        if isinstance(mesh, trimesh.points.PointCloud):
            print(f"Mesh at {mesh_obj} is a PointCloud, skipping.")
            return None

        mesh.update_faces(mesh.unique_faces())
        mesh.merge_vertices()
        
        
        vertices = np.array(mesh.vertices)
        if up_dir == "y":
            vertices[:, [1, 2]] = vertices[:, [2, 1]]
        elif up_dir == "x":
            vertices[:, [0, 2]] = vertices[:, [2, 0]]

        vertices[:, 2] -= np.min(vertices[:, 2]) # align bottom to z=0
        faces = np.array(mesh.faces).astype(np.int32)

        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            print(f"Mesh at {mesh_obj} has no vertices or faces, skipping.")
            return None
        
        angle = get_sim_angles(vertices, faces)
        return angle
    except Exception as e:
        print(f"Error processing {mesh_obj}: {e}")
        return 

def simulate_scenes(mesh_scene_paths, up_dir="y", num_workers=None):
    try:
        angles = []
        for mesh_path in tqdm(mesh_scene_paths):
            if not os.path.exists(mesh_path):
                print(f"Mesh path {mesh_path} does not exist, skipping.")
                angles.append(None)
            
            mesh_scene = trimesh.load(mesh_path)
            current_angles = []
            for geom in mesh_scene.geometry.values():
                angle = simulate_single_model_mesh_obj(geom, up_dir)
                current_angles.append(angle)
            angles.append(current_angles)
        return angles
    except Exception as e:
        print(f"Error in simulate_multiple_models: {e}")
        return None