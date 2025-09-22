import numpy

bbox_min = numpy.array([0.0, 0.0, 0.0])
bbox_max = numpy.array([1.0, 1.0, 1.0])
octree_depth = 2
num_cells = numpy.exp2(octree_depth) # 8
x = numpy.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=numpy.float32)
y = numpy.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=numpy.float32)
z = numpy.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=numpy.float32)

matrix = numpy.meshgrid(x, y, z, indexing="ij")
xyz = numpy.stack((matrix[0], matrix[1], matrix[2]), axis=-1)
print(xyz.shape) # (5, 5, 5, 3)
xyz = xyz.reshape(-1, 3)
print(xyz.shape) # (125, 3)
print(xyz)