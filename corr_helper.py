import numpy as np
from scipy import sparse
from vertex_formulation import *
import click


def create_smoothness_identity_matrix(mesh, weight_s, weight_i, marker):
    transformations = [get_transformation_matrix(mesh, face) for face in range(mesh.num_faces)]
    A_marker, b_marker = get_marker_matrix(mesh, marker)
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


def get_marker_matrix(mesh, marker):
    A_marker = sparse.dok_matrix((marker.shape[0]*3, 3*(mesh.num_vertices + mesh.num_faces)))
    b_marker = np.zeros((A_marker.shape[0], ))
    for i, m in enumerate(marker):
        idx = 3*m[0]
        target_position = m[1]
        for j in range(3):
            A_marker[3*i + j, idx + j] = 1000000
            b_marker[3*i + j] = 1000000*target_position[j]
    return A_marker.tocsr(), b_marker


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
