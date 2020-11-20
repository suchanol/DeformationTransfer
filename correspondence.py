import pymesh
import closest_point
from scipy.optimize import minimize, LinearConstraint
from corr_helper import *
from vertex_formulation import *

source_mesh = None
target_mesh = None


def main(source, target):
    # load meshes
    load_meshes(source, target)
    assert source_mesh is not None and target_mesh is not None

    markers = load_markers("horse_camel.cons")
    # tree = closest_point.create_tree(target_mesh)
    deformation_minimize(markers, source_mesh.vertices, [], 1.0, 0.001, 0)


def load_meshes(source, target):
    global source_mesh, target_mesh
    source_mesh = pymesh.load_mesh(source)
    target_mesh = pymesh.load_mesh(target)

    # enable mesh connectivity to get adjacency information for vertices/faces/voxels
    source_mesh.enable_connectivity()
    target_mesh.enable_connectivity()

    source_mesh.add_attribute("vertex_normal")
    target_mesh.add_attribute("face_normal")


def load_markers(file):
    m = []
    with open(file) as markers:
        markers.readline()
        for line in markers:
            vals = line.split(", ")
            vals[0] = int(vals[0])
            vals[1] = int(vals[1][:-1])
            m.append(vals)
    return np.array(m)


def conv_marker_to_constraint(markers):
    mat = np.zeros((source_mesh.num_vertices*3, source_mesh.num_vertices*3))
    for idx in markers:
        indices = [[3*idx[0]]*2, [3*idx[0] + 1]*2, [3*idx[0] + 2]*2]
        for i in indices:
            mat[i] = 1
    bound = np.zeros((source_mesh.num_vertices, 3))
    bound[markers[:, 0]] = target_mesh.vertices[markers[:, 1]]
    bound = bound.flatten()
    return LinearConstraint(mat, bound, bound)


def deformation_minimize(markers, vertices, cls_pts, weight_smoothness, weight_identity, weight_closest):
    vertex_guess = vertices
    cls_pts_guess = cls_pts

    def func_to_minimize(vtx):
        transformations = get_transformations(source_mesh, vtx)
        reshaped = vtx.reshape((source_mesh.num_vertices, 3))
        return weight_smoothness * deformation_smoothness(source_mesh, transformations, vtx) \
               + weight_identity * deformation_identity(source_mesh, transformations, vtx) \
               + weight_closest * deformation_closest(source_mesh, cls_pts)

    constraints = conv_marker_to_constraint(markers)
    result = minimize(func_to_minimize, vertex_guess.flatten(), constraints=constraints)
    if result.success:
        source_mesh.vertices = result.x.reshape((source_mesh.vertices, 3))
        source_mesh.save_mesh("source_mesh_deformed.obj")


if __name__ == '__main__':
    main("horse-01.obj", "camel-01.obj")
