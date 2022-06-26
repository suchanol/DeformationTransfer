import numpy as np


def triangle_to_transformation_matrix(triangles):
    v = np.linalg.inv(triangle_to_matrix(triangles)).transpose(0, 2, 1)
    return np.c_[-np.sum(v, axis=2)[:, :, np.newaxis], v]


def triangle_to_transformation(src_tri, dest_tri):
    v = triangle_to_matrix(src_tri)
    v_tilde = triangle_to_matrix(dest_tri)
    return v_tilde @ np.linalg.inv(v)


def calc_normal(triangles):
    v1p = triangles[:, 0]
    v2p = triangles[:, 1]
    v3p = triangles[:, 2]
    normal = np.cross(v2p - v1p, v3p - v1p)
    normal = normal / np.sqrt(np.linalg.norm(normal, axis=-1))[:, np.newaxis]
    return normal


def triangle_to_matrix(triangles):
    v1p = triangles[:, 0]
    v2p = triangles[:, 1]
    v3p = triangles[:, 2]
    normal = calc_normal(triangles)
    return np.array([v2p.T - v1p.T, v3p.T - v1p.T, normal.T]).T


# just testing
if __name__ == '__main__':
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    t1 = np.array([p1, p2, p3])
    triangles = [t1]
    for _ in range(10):
        triangles = np.append(triangles, [t1], axis=0)
    triangles = np.append(triangles, np.array([[p1, 2*p2, 2*p3]]), axis=0)
    V = np.arange(9).reshape(3, 3)
    print(triangles)

    m = triangle_to_matrix(triangles)
    print(np.linalg.inv(m))
    print(triangle_to_transformation(triangles, triangles))
    print(triangle_to_transformation_matrix(triangles))
    pass