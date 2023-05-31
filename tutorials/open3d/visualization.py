import os
import open3d as o3d


def visualization(filename):
    # Load a point cloud file.
    pcd = o3d.io.read_point_cloud(filename)

    # This function takes a list of geometry objects, and renders them together.
    window_name = "Visualization — " + os.path.basename(filename)
    o3d.visualization.draw_geometries([pcd], window_name, width=1000, height=800, left=200, top=150)


if __name__ == "__main__":
    input_filename = "data/input/bunny.ply"
    visualization(input_filename)
