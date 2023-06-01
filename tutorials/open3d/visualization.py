import os
import sys

import numpy as np
import open3d as o3d


def visualization(filename, show_coordinate=False):
    geometries = []

    # Load a point cloud file
    pcd = o3d.io.read_point_cloud(filename)
    geometries.append(pcd)

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    mean = np.mean(max_bound - min_bound)

    if show_coordinate:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=mean * 0.2, origin=min_bound - mean * 0.1)
        geometries.append(coordinate)

    # This function takes a list of geometry objects, and renders them together
    o3d.visualization.draw_geometries(geometries, window_name="Visualization — {}".format(os.path.basename(filename)),
                                      width=1000, height=800, left=400, top=150)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        filenames.append("data/input/bunny.ply")

    for index, input_filename in enumerate(filenames):
        print(f"Visualize file [{index + 1}]: {input_filename}")
        visualization(input_filename, show_coordinate=True)
