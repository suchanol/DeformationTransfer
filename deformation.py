import pymesh
import scipy.sparse.linalg
from vertex_formulation import *


# find out which triangle in the mesh shares the vertex of index vi
# return the index of the faces
def find_triangle_fan(vi, mesh):
    return mesh.get_vertex_adjacent_faces(vi)


# load the animation .obj files into a list
def load_deformation(n, name):
    result = []
    # for i in range(1, n + 1):
    # result.append(pymesh.load_mesh("{}-{:02d}.obj".format(name, i)))
    return pymesh.load_mesh(name)
    # return result


# save the obj in the list to .obj files
def save_deformation(result, name):
    n = len(result)
    for i in range(0, n):
        pymesh.save_mesh("{}-{:02d}.obj".format(name, i), result[i])


def find_source_trans_matrix(mesh, deform):
    result = get_transformations(mesh, deform.vertices)
    return result


def find_triangle_matrix(mesh):
    # B = V^(-t) in diagonal form
    # 3 equal B for every triangle
    V = [(triangle3_to_matrix(mesh.vertices[triangle])) for triangle in mesh.faces]
    V_t = [(v.transpose()) for v in V]
    V_t_inv = [(np.linalg.inv(v_t)) for v_t in V_t]
    # return V_t_inv
    bs = []
    for b in V_t_inv:
        for i in range(0, 3):
            bs.append(b)
    B = scipy.linalg.block_diag(*bs)
    return B


def find_vertex_correlation_matrix(mesh):
    points = [t for t in mesh.vertices]
    vertices = []
    for triangle in mesh.faces:
        t_4 = calc_normal(mesh.vertices[triangle])
        # vertices.append(t_4[0])
        # vertices.append(t_4[1])
        # vertices.append(t_4[2])
        # vertices.append(t_4[3])
        vertices.append(t_4)
        points.append(t_4[3])
    D = np.zeros([4 * 3 * mesh.num_faces, 3 * len(points)])

    for i in range(0, len(points)):
        for p in points[i]:
            for j in range(0, len(vertices)):
                if point_is_present(points[i], vertices[j]):
                    D[4*3*j][3*i] = 1
                    D[(4*3*j)+1][(3*i)+1] = 1
                    D[(4*3*j)+2][(3*i)+2] = 1
    return D

def point_is_present(point, vertices):
    for v in vertices:
        if np.array_equal(point, v):
            return True
    return False

def find_neighbor(List, vi):
    result = []
    for triangle in List:
        for i in triangle:
            if vi == i:
                result.append(triangle)
                break
    return result


def find_triangle(List, t):
    result = []
    for i in range(0, len(List)):
        if np.array_equal(t, List[i]):
            result.append(i)
    return result


# the original matrix uses v1 v2 v3 which are differences of points, now function inverses the process
def find_vector_to_point_matrix(n):
    m = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    # return scipy.linalg.block_diag(m,m,m)
    ms = []
    for i in range(0, n):
        ms.append(m)
    # return ms
    M = scipy.linalg.block_diag(*ms)
    return M

# the vertices are given in the form of (v1x, v2x, v3x, v1y......v4z) for each triangle change it to (v1x, v2x, v3x,
# v4x...)
def find_realignment_Q_tilde_matrix(n):
    lines = []
    for i in range(0, 12):
        line = np.zeros(12)
        if i % 3 == 0:
            line[i // 3] = 1
        if i % 3 == 1:
            line[(i // 3) + 4] = 1
        if i % 3 == 2:
            line[(i // 3) + 8] = 1
        lines.append(line)
    q = np.array(lines)
    # return q
    qs = []
    for i in range(0, n):
        qs.append(q)
    Q = scipy.linalg.block_diag(*qs)
    return Q


# the vertices are given in the form of (v1x, v1y, v1z, v2x......v4z) for each triangle change it to (v1x, v2x, v3x,
# v4x...)
def find_realignment_Q_matrix(n):
    lines = []
    for i in range(0, 9):
        line = np.zeros(9)
        if i < 3:
            line[i * 3] = 1
        if 3 <= i < 6:
            line[1 + ((i * 3) % 9)] = 1
        if 6 <= i < 9:
            line[2 + ((i * 3) % 9)] = 1
        lines.append(line)
    q = np.array(lines)
    # return q
    qs = []
    for i in range(0, n):
        qs.append(q)
    Q = scipy.linalg.block_diag(*qs)
    return Q

def BQMQ_multiplication(B,Q,M,Q_tilde):
    result = []
    for i in range (0, len(B)):
        B_3 = scipy.linalg.block_diag(B[i],B[i],B[i])
        result.append(B_3@Q@M@Q_tilde)

    return result
def find_target_trans_matrix(mesh):
    # print(mesh.num_faces)
    B_matrix = find_triangle_matrix(mesh)
    # print(B_matrix)
    Q_matrix = find_realignment_Q_matrix(mesh.num_faces)
    # print(Q_matrix)
    M_matrix = find_vector_to_point_matrix(mesh.num_faces*3)
    # print(M_matrix)
    Q_tilde_matrix = find_realignment_Q_tilde_matrix(mesh.num_faces)
    # print(Q_tilde_matrix)
    D_matrix = find_vertex_correlation_matrix(mesh)
    # print(D_matrix)
    result = B_matrix @ Q_matrix @ M_matrix @ Q_tilde_matrix @ D_matrix
    return result

    # to be calculated |S - T| where T is given by a combination of matrices B Q M D P Q^
    # T = BQMQ^DP
    # P is the vector of the unknown points of the transformed target mesh
    # Q is a re-arrangement matrix to change the order of the unknown ( to facilitate calculation only) same for Q^(-1)
    # D is the matrix of vertices correlation
    # B is the matrix of the target mesh vectors
    # M is the transformation of the vector coordinate in the original point coordinate

    # B = find_triangle_matrix(mesh)
    # Q = find_realignment_Q_matrix(mesh.num_faces)
    # M = find_vector_to_point_matrix(mesh.num_faces*3)
    # Q_tilde = find_realignment_Q_tilde_matrix(mesh.num_faces)
    # BQMQ_tilde = BQMQ_multiplication(B,Q,M,Q_tilde)
    # BQMQ_matrix = scipy.linalg.block_diag(*BQMQ_tilde)
    # D_matrix = find_vertex_correlation_matrix(mesh)
    # return BQMQ_matrix@D_matrix

def main(source, deform_s, target):
    n, deform_name = deform_s
    source_deforms = load_deformation(n, deform_name)
    source_mesh = pymesh.load_mesh(source)
    target_mesh = pymesh.load_mesh(target)
    source_deform_matrix = find_source_trans_matrix(source_mesh, source_deforms)
    source_deform_vector = np.concatenate(source_deform_matrix, axis=None)
    target_deform_matrix = find_target_trans_matrix(target_mesh)
    x = scipy.sparse.linalg.lsqr(target_deform_matrix, source_deform_vector, show=True)
    print(x)
    return


if __name__ == '__main__':
    main("horse_ref_200_faces.obj", (2, "horse-gallop-01_200_faces.obj"), "camel_ref_200_faces.obj")
