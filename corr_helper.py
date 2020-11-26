import numpy as np
from scipy import sparse, linalg
from vertex_formulation import *
import click


def get_transformation_matrix(mesh, face):
    triangle = mesh.vertices[mesh.faces[face]]
    mat_v = inv(triangle3_to_matrix(triangle))
    block = np.append([[-np.sum(mat_v[:, i])] for i in range(3)], mat_v, axis=1)
    return block


def create_smoothness_identity_matrix(mesh, weight_s, weight_i, marker_constraints):
    # A = sparse.csr_matrix((mesh.num_faces*9*3, 3*(mesh.num_vertices + mesh.num_faces)))
    A_smoothness = []
    A_identity = []
    c_smoothness = []
    c_identity = []
    transformations = [get_transformation_matrix(mesh, face) for face in range(mesh.num_faces)]
    with click.progressbar(range(mesh.num_faces)) as progressbar:
        for face in progressbar:
            face_matrix, face_c = get_transformation_term(mesh, face, transformations[face], marker_constraints)
            A_sub_smoothness, c_sub_smoothness = get_smoothness_matrix(mesh, face_matrix, face_c, face, transformations, marker_constraints)
            A_smoothness.append(weight_s * A_sub_smoothness)
            c_smoothness.append(c_sub_smoothness)

            A_sub_identity, c_sub_identity = get_identity_matrix(face_matrix, face_c)
            A_identity.append(weight_i * A_sub_identity)
            c_identity.append(c_sub_identity)

    A = sparse.vstack([sparse.vstack(A_smoothness), sparse.vstack(A_identity)])
    c = np.hstack([np.hstack(c_smoothness), np.hstack(c_identity)])
    return A, c


def get_identity_matrix(face_matrix, face_c):
    return face_matrix, face_c - np.identity(3).flatten()


def get_smoothness_matrix(mesh, face_matrix, face_c, face_idx, transformations, marker_constraints):
    A_smoothness = []
    c_smoothness = []
    adjacent_faces = mesh.get_face_adjacent_faces(face_idx)
    for adj_face_idx in adjacent_faces:
        transformation = transformations[adj_face_idx]
        adj_face_matrix, c_adj_face = get_transformation_term(mesh, adj_face_idx, transformation, marker_constraints)
        A_smoothness.append(face_matrix - adj_face_matrix)
        c_smoothness.append(c_adj_face)

    if len(A_smoothness) == 0:
        return sparse.csr_matrix((3*9, 3*(mesh.num_vertices + mesh.num_faces))), np.zeros((9*3,))
    return sparse.vstack(A_smoothness), np.hstack(c_smoothness)


def get_transformation_term(mesh, face, transformation, marker_constraints):
    c = np.zeros((9,))
    rows = []
    cols = []
    data = []
    corners = mesh.faces[face]
    for row in range(3):
        for i in range(3):
            for j in range(4):
                rows.append(row*3 + i)
                coefficient = transformation[i, j]
                data.append(coefficient)
                if j < 3:
                    cols.append(3*corners[j] + row)

                    if corners[j] in marker_constraints[:, 0]:
                        target_position = marker_constraints[np.where(marker_constraints[:, 0] == corners[j]), 1][0][0]
                        c[i * 3 + j] = c[i*3 + j] - coefficient * target_position[row]
                else:
                    cols.append(mesh.num_vertices*3 + 3*face + row)

    return sparse.csr_matrix((data, (rows, cols)), shape=(9, 3*(mesh.num_vertices + mesh.num_faces))), c


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
