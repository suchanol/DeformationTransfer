import numpy as np
from scipy import sparse
from vertex_formulation import *
import click


def create_full_matrix(mesh, weight_s, weight_i, weight_c, closest_points, marker):
    A_s_i, b_s_i = create_smoothness_identity_matrix(mesh, weight_s, weight_i, marker)

    A_closest = sparse.hstack([weight_c * sparse.identity((3*mesh.num_vertices)), sparse.csc_matrix((3*mesh.num_vertices, 3*mesh.num_faces))])
    b_closest = weight_c * closest_points
    return sparse.vstack([A_s_i, A_closest]), np.hstack([b_s_i, b_closest])


def create_smoothness_identity_matrix(mesh, weight_s, weight_i, marker):
    transformations = [get_transformation_matrix(mesh, face) for face in range(mesh.num_faces)]
    A_smoothness = sparse.dok_matrix((mesh.num_faces*3*9, 3*(mesh.num_vertices + mesh.num_faces)))
    b_smoothness = np.zeros((mesh.num_faces * 9 * 3,))
    A_identity = sparse.dok_matrix((mesh.num_faces*9, 3*(mesh.num_vertices + mesh.num_faces)))
    b_identity = np.zeros((mesh.num_faces * 9, ))
    with click.progressbar(range(mesh.num_faces)) as progressbar:
        for face in progressbar:
            set_smoothness_term(A_smoothness, b_smoothness, mesh, face, transformations, weight_s, marker)
            set_identity_term(A_identity, b_identity, mesh, face, transformations, weight_i, marker)

    A = sparse.vstack([A_smoothness.tocsr(), A_identity.tocsr()])
    b = np.hstack([b_smoothness, b_identity])
    return A, b


def set_smoothness_term(A, b, mesh, face, transformations, weight_s, marker):
    adj_faces = mesh.get_face_adjacent_faces(face)
    rows, cols, data, c = get_transformation_term_indices(mesh, face, transformations[face], marker)
    rows = rows + (face*3*9)
    data = weight_s * data
    for i, adj in enumerate(adj_faces):
        adj_rows, adj_cols, adj_data, adj_c = get_transformation_term_indices(mesh, adj, transformations[adj], marker)
        adj_rows = adj_rows + (face*3*9 + i*9)
        A[rows, cols] = data
        A[adj_rows, adj_cols] = A[adj_rows, adj_cols] - weight_s * adj_data
        b[face*9*3 + i*9:face*9*3 + (i+1)*9] = weight_s * (c - adj_c)
        rows = rows + 9


def set_identity_term(A, b, mesh, face, transformations, weight_i, marker):
    rows, cols, data, c = get_transformation_term_indices(mesh, face, transformations[face], marker)
    rows = rows + (face*9)
    A[rows, cols] = weight_i * data
    b[face*9:(face+1)*9] = b[face*9:(face+1)*9] + weight_i * (c + np.identity(3).flatten())
