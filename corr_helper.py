import numpy as np
from scipy.linalg import *


# is minimized when the change in deformation (and not the mesh itself) is smooth
def deformation_smoothness(mesh, transformations, vertices):
    # for each triangle we need a transformation
    #assert mesh.num_faces == transformations.size

    # calculate squared Frobenius norm of T_i and T_j (where T_j is adjacent to T_i; ||T_i - T_j||_F^2)
    def calc_norm(triangle, adj):
        return norm(transformations[triangle] - transformations[adj]) ** 2

    # calculate sum of calc_norm over all triangles adjacent to T_i (\sum_{j \in adj(i)} calc_norm(i, j))
    def adj_sum_norm(triangle):
        return np.sum([calc_norm(triangle, adjacent) for adjacent in mesh.get_face_adjacent_faces(tri)])

    # calculate sum of all adj_sum_norm over all triangles of the mesh (\sum_{i \in faces} adj_sum_norm(i))
    return np.sum([adj_sum_norm(triangle) for triangle in range(mesh.num_faces)])


# is minimized when all transformations are equal to the identity matrix
def deformation_identity(mesh, transformations, vertices):
    # for each triangle we need a transformation
    #assert mesh.num_faces == transformations.size

    # return sum of squared Frobenius-distance of each triangle transformation to the identity
    # (\sum_{i in faces} ||T_i - I||_F^2)
    return np.sum([norm(transformations[triangle] - np.identity(3))**2 for triangle in range(mesh.num_faces)])


def deformation_closest(mesh, closest_points):
    # for each vertex we need a corresponding closest valid point
    assert mesh.num_vertices == closest_points.size

    # return sum of squared distances of target vertex to its closest valid point on source mesh
    # (\sum_{i in vertices} ||v_i - c_i||^2)
    return np.sum([norm(vertex - closest_points[vert_index])**2 for vert_index, vertex in enumerate(mesh.vertices)])
