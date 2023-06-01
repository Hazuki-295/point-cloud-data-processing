import os
import sys

import open3d as o3d


def visualization(filename):
    # Load a point cloud file.
    pcd = o3d.io.read_point_cloud(filename)

    # This function takes a list of geometry objects, and renders them together.
    o3d.visualization.draw_geometries([pcd], window_name="Visualization — {}".format(os.path.basename(filename)),
                                      width=1000, height=800, left=200, top=150)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("No input filenames passed.\n")
        filenames.append("data/input/bunny.ply")

    for index, input_filename in enumerate(filenames):
        print(f"Visualize file [{index + 1}]: {input_filename}")
        visualization(input_filename)
