import numpy as np
from scipy.linalg import inv, norm, block_diag
from scipy import sparse


def get_transformation_term_indices(mesh, face, transformation, marker_constraints):
    c = np.zeros((9, ))
    rows = []
    cols = []
    data = []
    corners = mesh.faces[face]
    for row in range(3):
        for i in range(3):
            for j in range(4):
                coefficient = transformation[i, j]
                if j < 3:
                    if corners[j] in marker_constraints[:, 0]:
                        target_position = marker_constraints[np.where(marker_constraints[:, 0] == corners[j])][0, 1]
                        c[row * 3 + i] = c[row * 3 + i] - coefficient * target_position[row]
                        continue
                    cols.append(3 * corners[j] + row)
                else:
                    cols.append(mesh.num_vertices*3 + 3*face + row)
                rows.append(row * 3 + i)
                data.append(coefficient)

    return np.array(rows), np.array(cols), np.array(data), c


def get_transformation_matrix(mesh, face):
    triangle = mesh.vertices[mesh.faces[face]]
    mat_v = inv(triangle3_to_matrix(triangle))
    block = np.append([[-np.sum(mat_v[:, i])] for i in range(3)], mat_v.T, axis=1)

    assert np.allclose(triangle3_to_matrix(triangle) @ mat_v, np.identity(3))
    assert np.allclose(block_diag(block, block, block).dot(np.array(calc_normal(triangle)).flatten("F")), np.identity(3).flatten())

    return block


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


def triangle3_to_matrix(triangle):
    triangle_with_normal = calc_normal(triangle)
    return triangle_to_matrix(triangle_with_normal)


def triangle_to_matrix(triangle):
    v_1, v_2, v_3, v_4 = triangle
    return np.array([v_2 - v_1, v_3 - v_1, v_4 - v_1]).T


def get_transformations(mesh, vertices):
    transformation = [triangle_to_transformation(mesh.vertices[triangle], vertices[triangle]) for triangle in mesh.faces]
    return transformation
