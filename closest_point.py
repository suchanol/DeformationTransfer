# auxiliary module to calculate the closest valid point on a vertex to a specific point
import numpy as np
from itertools import count
from scipy.spatial import cKDTree


# generate a KDTree according to the mesh
def create_tree(mesh):
    return cKDTree(mesh.vertices)


# query the spatial tree structure for the nearest valid neighbor and returns the vertex' index
def get_closest_valid_point(tree, source_mesh, target_mesh, vertex_idx):
    assert tree is not None

    vertex_normal = source_mesh.get_vertex_attribute("vertex_normal")[vertex_idx]
    vertex = source_mesh.vertices[vertex_idx]
    for index in count(1):
        dist, idx = tree.query(vertex, k=index)
        if index > 1:
            idx = idx[index - 1]
        if idx >= target_mesh.num_vertices:
            return 0
        adj_face_normals = target_mesh.get_face_attribute("face_normal")[target_mesh.get_vertex_adjacent_faces(idx)]
        if is_valid(vertex_normal, *adj_face_normals):
            return idx


# check if a point is valid: normal at point differs in orientation to adjacent triangles of target point
# less than 90 degrees
def is_valid(vertex_normal, *triangle_normals):
    valid = True
    for normal in triangle_normals:
        valid = valid & (np.dot(vertex_normal, normal) > 0)
    return valid


# get closest valid points for all vertices of a source_mesh
def get_closest_valid_points(tree, source_mesh, target_mesh):
    return [get_closest_valid_point(tree, source_mesh, target_mesh, idx) for idx in range(source_mesh.num_vertices)]


