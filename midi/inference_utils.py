from typing import List, Tuple

import numpy as np
import PIL
import torch.nn.functional as F
from PIL import Image


def generate_dense_grid_points(
    bbox_min: np.ndarray, bbox_max: np.ndarray, octree_depth: int, indexing: str = "ij"
):
    '''
    Generate dense grid points in a 3D bounding box defined by bbox_min and bbox_max.
    The number of grid points along each dimension is determined by octree_depth (2^octree_depth + 1).
    Args:
        bbox_min: np.ndarray of shape (3,), minimum coordinates of the bounding box.
        bbox_max: np.ndarray of shape (3,), maximum coordinates of the bounding box.
        octree_depth: int, depth of the octree, determines the number of grid points.
        indexing: str, 'ij' for matrix indexing, 'xy' for Cartesian indexing.      
    Returns:
        xyz: np.ndarray of shape (num_points, 3), the generated grid points.
        grid_size: List[int] of length 3, number of points along each dimension.
        length: np.ndarray of shape (3,), length of the bounding box along each dimension.
    '''
    # 3D coordinate [x_min, y_min, z_min]: bbox_min
    # 3D coordinate [x_max, y_max, z_max]: bbox_max
    # octree_depth: int, e.g. 4, 5, 6
    length = bbox_max - bbox_min # length in each dimension
    num_cells = np.exp2(octree_depth) # number of cells in each dimension: 2^octree_depth
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing) # 3D matrix of shape (num_cells+1, num_cells+1, num_cells+1)
    xyz = np.stack((xs, ys, zs), axis=-1) # stack: stack 3 matrices to 1 matrix of shape (num_cells+1, num_cells+1, num_cells+1, 3)
    xyz = xyz.reshape(-1, 3) # flatten to (num_points, 3) # 3 for x, y, z
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length
