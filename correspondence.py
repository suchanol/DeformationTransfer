import pymesh
import closest_point
from corr_helper import *
from vertex_formulation import *
from timeit import default_timer as timer
import scipy.sparse.linalg
import corres_resolve

source_mesh = None
target_mesh = None


def main(source, target):
    # load meshes
    load_meshes(source, target)
    assert source_mesh is not None and target_mesh is not None

    markers = load_markers("default.cons")

    tree = closest_point.create_tree(target_mesh)
    deformed_mesh = solve_correspondence_problem(tree, markers)
    corres_resolve.compute_pairs("face.corres", deformed_mesh, target_mesh)


def solve_correspondence_problem(tree, markers):
    weight_s = 1.0
    weight_i = 0.001

    A, b = create_smoothness_identity_matrix(source_mesh, weight_s, weight_i, markers)
    deformed_mesh = minimize(A, b, markers)

    for weight_c in np.linspace(1.0, 5000.0, 4):
        closest_points = target_mesh.vertices[closest_point.get_closest_valid_points(tree, deformed_mesh, target_mesh)].flatten()
        closest_points = set_marker_positions(closest_points, markers)
        A_full, b_full = create_full_matrix(source_mesh, weight_s, weight_i, weight_c, closest_points, markers)

        deformed_mesh = minimize(A_full, b_full, markers)

    deformed_mesh.add_attribute("face_centroid")
    deformed_mesh.add_attribute("face_normal")

    return deformed_mesh


def minimize(A, b, markers, save_mesh=True):
    start = timer()
    x_0 = np.append(source_mesh.vertices.flatten(),
                    [calc_normal(source_mesh.vertices[source_mesh.faces[triangle]])[3]
                    for triangle in range(source_mesh.num_faces)])
    x_0 = set_marker_positions(x_0, markers)

    x = sparse.linalg.lsqr(A, b, x0=x_0, show=True)[0]
    x = set_marker_positions(x, markers)

    end = timer()
    print(end - start)

    deformed_mesh = pymesh.form_mesh(x[:source_mesh.num_vertices * 3].reshape((source_mesh.num_vertices, 3)),
                                     source_mesh.faces, source_mesh.voxels)
    deformed_mesh.add_attribute("vertex_normal")
    deformed_mesh.enable_connectivity()

    if save_mesh:
        pymesh.save_mesh("source_mesh_deformed.obj", deformed_mesh)

    return deformed_mesh


def load_meshes(source, target):
    global source_mesh, target_mesh
    source_mesh = pymesh.load_mesh(source)
    target_mesh = pymesh.load_mesh(target)

    # enable mesh connectivity to get adjacency information for vertices/faces/voxels
    source_mesh.enable_connectivity()
    target_mesh.enable_connectivity()

    source_mesh.add_attribute("vertex_normal")

    target_mesh.add_attribute("face_normal")
    target_mesh.add_attribute("face_centroid")


def set_marker_positions(x, marker):
    y = x.copy()
    for m in marker:
        idx = 3 * m[0]
        target_position = m[1]
        for j in range(3):
            y[idx + j] = target_position[j]
    return y


def load_markers(file):
    m = []
    with open(file) as markers:
        markers.readline()
        for line in markers:
            vals = line.split(", ")
            vals[0] = int(vals[0])
            vals[1] = target_mesh.vertices[int(vals[1][:-1])]
            m.append(vals)
    return np.array(m)


if __name__ == '__main__':
    main("horse_ref_decimate.obj", "camel_ref_decimate.obj")
