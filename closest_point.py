# auxiliary module to calculate the closest valid point on a vertex to a specific point
import numpy as np
from itertools import count
from scipy.spatial import KDTree

# generate a KDTree according to the mesh
def create_tree(mesh):
    return KDTree(mesh.vertices)


# get closest valid points for all vertices of a source_mesh
def get_closest_valid_points(tree, source_mesh, target_mesh):
    queries = tree.query(source_mesh.vertices, k=50)
    source_vertex_normals = source_mesh.get_vertex_attribute("vertex_normal")[np.arange(source_mesh.num_vertices)]
    target_vertex_normals = target_mesh.get_vertex_attribute("vertex_normal")[queries[1]]
    mask = np.empty(queries[1].shape)
    for idx in np.arange(source_mesh.num_vertices):
        mask[idx] = np.arccos(np.inner(source_vertex_normals[idx], target_vertex_normals[idx])) < np.radians(90)

    return queries[1][np.arange(source_mesh.num_vertices), np.argmax(mask, axis=1)]


