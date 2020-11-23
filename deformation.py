import pymesh
from vertex_formulation import *


# find out which triangle in the mesh shares the vertex of index vi
# return the index of the faces
def find_triangle_fan(vi, mesh):
    return mesh.get_vertex_adjacent_faces(vi)


# load the animation .obj files into a list
def load_deformation(n, name):
    result = []
    for i in range(1, n + 1):
        result.append(pymesh.load_mesh("{}-{:02d}.obj".format(name, i)))
    return result


# save the obj in the list to .obj files
def save_deformation(result, name):
    n = len(result)
    for i in range(0, n):
        pymesh.save_mesh("{}-{:02d}.obj".format(name, i), result[i])


def main(source, deform_s):
    n, deform_name = deform_s
    source_deform = load_deformation(n, deform_name)
    source_mesh = pymesh.load_mesh(source)
    source_deform_matrices = []
    i = 0
    for s in source_deform:
        source_deform_matrices.append(get_transformations(source_mesh, s.vertices))
        i += 1
        print(i)
    return


if __name__ == '__main__':
    main("horse-01.obj", (48, "res/horse-gallop/horse-gallop"))
