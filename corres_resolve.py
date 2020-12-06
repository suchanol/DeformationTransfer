from scipy.linalg import norm
from itertools import count
from scipy.spatial import cKDTree

import numpy as np

threshold = 0.5


def compute_pairs(filename, source_mesh, target_mesh):
    source_tree = create_tree(source_mesh)
    target_tree = create_tree(target_mesh)

    source_pairs = get_closest_compatible_triangles(target_tree, source_mesh, target_mesh)
    target_pairs = get_closest_compatible_triangles(source_tree, target_mesh, source_mesh)
    target_pairs = [(b, a) for a, b in target_pairs]

    all_pairs = source_pairs + target_pairs

    with open(filename, "w") as f:
        for a, b in all_pairs:
            f.write("%i,%i\n" % (a, b))


def create_tree(mesh):
    return cKDTree(mesh.get_face_attribute("face_centroid"))


def get_closest_compatible_triangles(target_tree, source_mesh, target_mesh):
    return [(face, get_closest_compatible_triangle(target_tree, source_mesh, target_mesh, face))
            for face in range(source_mesh.num_faces)]


def get_closest_compatible_triangle(target_tree, source_mesh, target_mesh, face_idx):
    assert target_tree is not None

    face_centroid = source_mesh.get_face_attribute("face_centroid")[face_idx]
    for index in count(1):
        dist, idx = target_tree.query(face_centroid, k=index)
        if index > 1:
            idx = idx[index - 1]
        if idx >= target_mesh.num_faces:
            return 0
        if triangles_compatible(source_mesh, target_mesh, face_idx, idx):
            return idx


def triangles_compatible(source_mesh, target_mesh, source_face, target_face):
    source_centroid = source_mesh.get_face_attribute("face_centroid")[source_face]
    target_centroid = target_mesh.get_face_attribute("face_centroid")[target_face]

    source_normal = source_mesh.get_face_attribute("face_normal")[source_face]
    target_normal = target_mesh.get_face_attribute("face_normal")[target_face]

    if norm(source_centroid - target_centroid) <= threshold and np.dot(source_normal, target_normal) > 0:
        return True
    return False
