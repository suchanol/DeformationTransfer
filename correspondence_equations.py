import numpy as np
import vertex_formulation as vert
from scipy import sparse


def create_closest_points_matrix(mesh, vertices, closest_points, markers):
    #marker_indices = np.in1d(range(vertices.shape[0]), markers[:, 0].astype(int))
    Q = sparse.dok_matrix((3 * vertices.shape[0], 3 * (mesh.num_vertices + mesh.num_faces)))
    c = closest_points.flatten()

    Q[np.arange(Q.shape[0]), vertex_to_matrix_index(vertices).flatten()] = 1

    return Q, c


def create_identity_matrix(mesh, markers, triangles):
    Q, c = get_elementary_term(mesh, markers, triangles)
    c += np.broadcast_to(np.eye(3), (triangles.shape[0], 3, 3)).flatten()

    return Q, c


def create_smoothness_matrix(mesh, markers, triangles):
    adjacencies = np.array([], dtype=int)
    adj_amount = np.array([], dtype=int)
    for triangle in triangles:
        faces = mesh.get_face_adjacent_faces(triangle)
        adj_amount = np.append(adj_amount, faces.shape[0])
        adjacencies = np.append(adjacencies, faces)
    repeated_triangles = np.repeat(triangles, adj_amount)
    Q1, c1 = get_elementary_term(mesh, markers, repeated_triangles.flatten())
    Q2, c2 = get_elementary_term(mesh, markers, adjacencies.flatten())
    Q = Q1 - Q2
    c = c1 - c2

    return Q, c


def get_elementary_term(mesh, markers, triangles):
    triangle_span = mesh.faces[triangles]
    triangle_coords = mesh.vertices[triangle_span]
    transformations = vert.triangle_to_transformation_matrix(triangle_coords)
    Q = sparse.dok_matrix((9 * triangles.shape[0], 3 * (mesh.num_vertices + mesh.num_faces)))
    c = np.zeros((Q.shape[0],))

    vertex_indices = vertex_to_matrix_index(triangle_span.flatten())
    triangle_indices = triangle_to_matrix_index(mesh.num_vertices, triangles)
    transformation_indices = np.append(vertex_indices.reshape(-1, 3, 3),
                                       triangle_indices.reshape(-1, 1, 3), axis=1).transpose(0, 2, 1)
    transformation_indices = transformation_indices[:, :, :, np.newaxis] + np.array([0, 0, 0])[np.newaxis, :]
    col_indices = transformation_indices.transpose(0, 1, 3, 2).flatten()
    row_indices = (np.arange(Q.shape[0])[:, np.newaxis] + np.zeros((4,))[np.newaxis, :]).flatten()
    transformations = transformations[:, np.newaxis] + np.zeros((3, 3, 4))[np.newaxis, :]
    Q[row_indices, col_indices] = transformations.flatten()

    # filter out marked vertices
    if markers.size != 0:
        for i, marker in enumerate(markers[:, 0]):
            marker_indices = np.where(triangle_span == marker)[0]
            marker_rows = (marker_indices * 9)[:, np.newaxis] + np.arange(9)
            marker_cols = vertex_to_matrix_index(np.array([marker]))
            marker_cols = marker_cols[:, :, np.newaxis] + np.zeros((3,))[np.newaxis, :]
            marker_cols = np.broadcast_to(marker_cols, (marker_indices.shape[0], *marker_cols.shape))
            marker_rows = marker_rows.astype(np.int).flatten()
            marker_cols = marker_cols.astype(np.int).flatten()

            c[marker_rows] -= Q[marker_rows, marker_cols].toarray().flatten() * \
                              np.broadcast_to(np.repeat(markers[i, 1], 3), (marker_indices.shape[0], 9)).flatten()
            Q[marker_rows, marker_cols] = 0

    return Q, c


def vertex_to_matrix_index(vertices):
    return vertices[:, np.newaxis]*3 + np.arange(3)[np.newaxis, :]


def triangle_to_matrix_index(num_vertices, triangles):
    return triangles[:, np.newaxis]*3 + (np.arange(3) + 3*num_vertices)[np.newaxis, :]


