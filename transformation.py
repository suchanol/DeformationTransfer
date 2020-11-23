import pymesh


# find out which triangle in the mesh shares the vertex of index vi
# return the index of the faces
def find_triangle_fan(vi, mesh):
    return mesh.get_vertex_adjacent_faces(vi)


def load_transformation(n, name):
    result = []
    for i in range(1, n + 1):
        print(("{}-{:02d}.obj".format(name, i)))
        result.append(pymesh.load_mesh("{}-{:02d}.obj".format(name, i)))
    return result


def main(source, trans_s, target):
    n, trans_name = trans_s
    source_trans = load_transformation(n, trans_name)
    return


if __name__ == '__main__':
    main("horse-01.obj", (48, "res/horse-gallop/horse-gallop"), "camel-01.obj")
