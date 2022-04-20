import correspondence_equations as eqns
import closest_point
import pymesh
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import vertex_formulation as vert
from functools import reduce
from timeit import default_timer as timer
import correspondence_resolve as resolve

def main(file_source_mesh, file_target_mesh, file_markers):
    # load meshes
    source_mesh = pymesh.load_mesh(file_source_mesh)
    target_mesh = pymesh.load_mesh(file_target_mesh)

    source_mesh.enable_connectivity()
    target_mesh.enable_connectivity()

    source_mesh.add_attribute("vertex_normal")

    target_mesh.add_attribute("vertex_normal")
    target_mesh.add_attribute("face_normal")
    target_mesh.add_attribute("face_centroid")

    markers = load_markers(file_markers, target_mesh)
    tree = closest_point.create_tree(target_mesh)
    start_time = timer()
    deformed_mesh = pymesh.load_mesh("full_deformed_mesh.obj")
    #deformed_mesh = solve_correspondence_problem(tree, source_mesh, target_mesh, markers)
    deformed_mesh.add_attribute("face_normal")
    deformed_mesh.add_attribute("face_centroid")

    print("compute correspondent pairs")
    pairs = resolve.compute_pairs("faces.corres", deformed_mesh, target_mesh)

    end_time = timer()
    print("Zeit", end_time - start_time)

def load_markers(file, target_mesh):
    m = []
    with open(file) as markers:
        markers.readline()
        for line in markers:
            vals = line.split(", ")
            vals[0] = int(vals[0])
            vals[1] = target_mesh.vertices[int(vals[1][:-1])]
            m.append(vals)
    return np.array(m)


def set_marker_positions(x, markers):
    new_x = x.copy()
    new_x[(markers[:, 0][:, np.newaxis]*3 + np.arange(3)[np.newaxis, :]).flatten().astype(int)] \
        = reduce(np.append, markers[:, 1])
    return new_x


def solve_correspondence_problem(tree, source_mesh, target_mesh, markers):
    print("start phase 1 of the correspondence problem")
    Q_smooth, c_smooth = eqns.create_smoothness_matrix(source_mesh, markers, np.arange(source_mesh.num_faces))
    Q_identity, c_identity = eqns.create_identity_matrix(source_mesh, markers, np.arange(source_mesh.num_faces))
    Q = sparse.vstack((1.0 * Q_smooth, 0.001 * Q_identity))
    c = np.hstack((1.0 * c_smooth, 0.001 * c_identity))

    """x0 = np.append(source_mesh.vertices.flatten(), vert.calc_normal(source_mesh.vertices[source_mesh.faces]))
    x0 = set_marker_positions(x0, markers)
    x = spla.lsqr(Q, c, x0=x0, show=True)[0]
    x = set_marker_positions(x, markers)

    deformed_mesh = pymesh.form_mesh(x[:source_mesh.num_vertices * 3].reshape((source_mesh.num_vertices, 3)),
                                     source_mesh.faces)
    deformed_mesh.enable_connectivity()
    deformed_mesh.add_attribute("vertex_normal")
    pymesh.save_mesh("predeformed_mesh.obj", deformed_mesh)"""
    deformed_mesh = pymesh.load_mesh("deformed_mesh.obj")
    deformed_mesh.add_attribute("vertex_normal")
    x = np.append(deformed_mesh.vertices.flatten(), vert.calc_normal(deformed_mesh.vertices[deformed_mesh.faces]))
    x = set_marker_positions(x, markers)

    print("start phase 2 of the correspondence problem")
    for weight_c in np.linspace(1.0, 5000.0, 4):
        print("closest points with weight_c: ", weight_c)
        closest_points = closest_point.get_closest_valid_points(tree, deformed_mesh, target_mesh)
        Q_close, c_close = eqns.create_closest_points_matrix(source_mesh, np.arange(source_mesh.num_vertices),
                                                             target_mesh.vertices[closest_points], markers)

        Q = sparse.vstack((Q, weight_c * Q_close))
        c = np.hstack((c, weight_c * c_close))
        x = spla.lsqr(Q, c, x0=x, show=True)[0]
        x = set_marker_positions(x, markers)

        deformed_mesh = pymesh.form_mesh(x[:source_mesh.num_vertices * 3].reshape((source_mesh.num_vertices, 3)),
                                         source_mesh.faces)
        deformed_mesh.enable_connectivity()
        deformed_mesh.add_attribute("vertex_normal")

    pymesh.save_mesh("full_deformed_mesh.obj", deformed_mesh)

    return deformed_mesh


if __name__ == '__main__':
    main("../DeformationTransfer/horse_ref.obj", "../DeformationTransfer/camel_ref.obj", "../DeformationTransfer/horse_camel.cons")