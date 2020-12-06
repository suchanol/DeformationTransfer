import pymesh
from corr_helper import *
from vertex_formulation import *
from timeit import default_timer as timer
import scipy.sparse.linalg

source_mesh = None
target_mesh = None


def main(source, target):
    # load meshes
    load_meshes(source, target)
    assert source_mesh is not None and target_mesh is not None

    markers = load_markers("default.cons")

    all_triangles = np.append(source_mesh.vertices.flatten(),
                              [calc_normal(source_mesh.vertices[source_mesh.faces[triangle]])[3]
                                for triangle in range(source_mesh.num_faces)])
    all_triangles = set_marker_positions(all_triangles, markers)
    A, c = create_smoothness_identity_matrix(source_mesh, 1.0, 0.001, markers)

    start = timer()
    x = sparse.linalg.lsqr(A, c, x0=all_triangles, show=True)[0]
    x = set_marker_positions(x, markers)

    end = timer()
    print(end - start)
    deformed_mesh = pymesh.form_mesh(x[:source_mesh.num_vertices*3].reshape((source_mesh.num_vertices, 3)), source_mesh.faces, source_mesh.voxels)
    pymesh.save_mesh("source_mesh_deformed.obj", deformed_mesh)
    # tree = closest_point.create_tree(target_mesh)


def load_meshes(source, target):
    global source_mesh, target_mesh
    source_mesh = pymesh.load_mesh(source)
    target_mesh = pymesh.load_mesh(target)

    # enable mesh connectivity to get adjacency information for vertices/faces/voxels
    source_mesh.enable_connectivity()
    target_mesh.enable_connectivity()

    source_mesh.add_attribute("vertex_normal")
    target_mesh.add_attribute("face_normal")


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
