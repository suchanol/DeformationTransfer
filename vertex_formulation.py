import numpy as np
from scipy.linalg import inv, norm


def calc_normal(triangle):
    v_1, v_2, v_3 = triangle
    normal = np.cross(v_2 - v_1, v_3 - v_1)
    v_4 = v_1 + normal / np.sqrt(norm(normal))
    return v_1, v_2, v_3, v_4


def triangle_to_transformation(orig_tri, deform_tri):
    orig = calc_normal(orig_tri)
    deformed = calc_normal(deform_tri)
    v = triangle_to_matrix(orig)
    v_tilde = triangle_to_matrix(deformed)
    return v_tilde * inv(v)


def triangle_to_matrix(triangle):
    v_1, v_2, v_3, v_4 = triangle
    return np.array([v_2 - v_1, v_3 - v_1, v_4 - v_1])


def get_transformations(mesh, vertices):
    transformation = [triangle_to_transformation(mesh.vertices[triangle], vertices[triangle]) for triangle in mesh.faces]
    return transformation
