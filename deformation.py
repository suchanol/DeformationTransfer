import pymesh
import scipy.linalg
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
    result.append(pymesh.load_mesh(name))
    return result


# save the obj in the list to .obj files
def save_deformation(result, name):
    n = len(result)
    for i in range(0, n):
        pymesh.save_mesh("{}-{:02d}.obj".format(name, i), result[i])


def find_source_trans_matrix(mesh, deform):
    result = get_transformations(mesh, deform.vertices)
    return result


def find_triangle_matrix(mesh):
    # B = V^(-t)
    V = [(triangle3_to_matrix(mesh.vertices[triangle])) for triangle in mesh.faces]
    V_t = [(v.transpose()) for v in V]
    V_t_inv = [(np.linalg.inv(v_t)) for v_t in V_t]
    V_t_inv_matrix = scipy.linalg.block_diag(*V_t_inv)
    Q_matrix = find_realignment_Q_matrix(len(V) // 3)
    M_matrix = find_vector_to_point_matrix(len(V))
    Q_tilde_matrix = find_realignment_Q_tilde_matrix(len(V) // 3)
    # result = BQM(Q^)DP
    result = V_t_inv_matrix @ Q_matrix @ M_matrix @ Q_tilde_matrix
    return result


# the original matrix uses v1 v2 v3 which are differences of points, now function inverses the process
def find_vector_to_point_matrix(n):
    m = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    ms = []
    for i in range(0, n):
        ms.append(m)
    M = scipy.linalg.block_diag(*ms)
    return M

def find_realignment_Q_tilde_matrix(n):
    lines = []
    for i in range(0, 12):
        line = np.zeros(12)
        if i % 3 == 0:
            line[i//3] = 1
        if i % 3 == 1:
            line[(i//3)+4] = 1
        if i % 3 == 2:
            line[(i//3)+8] = 1
        lines.append(line)
    q = np.array(lines)
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
    qs = []
    for i in range(0, n):
        qs.append(q)
    Q = scipy.linalg.block_diag(*qs)
    return Q


def find_target_trans_matrix(target_mesh):
    V_t_inv = find_triangle_matrix(target_mesh)

    # to be calculated |S - T| where T is given by a combination of matrices B Q M D P
    # T = BQMQ^(-1)DP
    # P is the vector of the unknown points of the transformed target mesh
    # Q is a re-arrangement matrix to change the order of the unknown ( to facilitate calculation only) same for Q^(-1)
    # D is the matrix of vertices correlation
    # B is the matrix of the target mesh vectors
    # M is the transformation of the vector coordinate in the original point coordinate


def main(source, deform_s, target):
    n, deform_name = deform_s
    source_deforms = load_deformation(n, deform_name)
    source_mesh = pymesh.load_mesh(source)
    target_mesh = pymesh.load_mesh(target)
    source_deform_matrix = find_source_trans_matrix(source_mesh, source_deforms[0])
    target_deform_matrix = find_target_trans_matrix(target_mesh)
    return


if __name__ == '__main__':
    main("horse_ref_decimate.obj", (2, "res/horse-gallop/horse-gallop-01_decimated.obj"), "camel_ref_decimate.obj")
