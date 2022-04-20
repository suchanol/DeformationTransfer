import numpy as np
import pymesh
import vertex_formulation as vert
import scipy.sparse.linalg as spla
import correspondence_equations as eqns

def main(file_source_mesh, file_target_mesh, *animations):
    source_mesh = pymesh.load_mesh(file_source_mesh)
    target_mesh = pymesh.load_mesh(file_target_mesh)

    corres_pairs = load_correspondences("faces.corres")
    print(corres_pairs)

    for animation in animations:
        cur_animation = pymesh.load_mesh(animation)
        source_face_coords = source_mesh.vertices[source_mesh.faces[corres_pairs[:, 0]]]
        cur_face_coords = cur_animation.vertices[cur_animation.faces[corres_pairs[:, 0]]]
        transformations = vert.triangle_to_transformation(source_face_coords, cur_face_coords)

        Q, _ = eqns.get_elementary_term(target_mesh, np.array([]), corres_pairs[:, 1])
        c = transformations.flatten()

        x0 = np.append(target_mesh.vertices.flatten(), vert.calc_normal(target_mesh.vertices[target_mesh.faces]))
        x = spla.lsqr(Q, c, x0=x0, show=True)[0]

        transferred_mesh = pymesh.form_mesh(x[:target_mesh.num_vertices * 3].reshape((target_mesh.num_vertices, 3)),
                                     target_mesh.faces)
        pymesh.save_mesh("new_nigger.obj", transferred_mesh)
        print(transformations)


def load_correspondences(filename):
    all_pairs = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            splitted = line.split(",")
            all_pairs.append([int(splitted[0]), int(splitted[1][:-1])])

    return np.array(all_pairs)

if __name__ == '__main__':
    main("../DeformationTransfer/horse_ref.obj", "../DeformationTransfer/camel_ref.obj", "../DeformationTransfer/horse-01.obj")