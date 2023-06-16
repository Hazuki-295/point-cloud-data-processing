import os
import sys

import open3d as o3d


def pipeline(filename):
    # Point cloud file I/O
    base_name, extension = os.path.splitext(os.path.basename(filename))
    output_path = os.path.join("data/output", base_name)
    os.makedirs(output_path, exist_ok=True)

    # Load the input point cloud file
    pcd = o3d.t.io.read_point_cloud(filename)

    # Down sample, evenly select 1 point for every k points
    down_sampled_pcd = pcd.uniform_down_sample(every_k_points=10)

    # Outlier removal
    filtered_pcd, _ = down_sampled_pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        exit(-1)

    for index, input_filename in enumerate(filenames):
        print(f"Point cloud processing [{index + 1}]: {input_filename}")
        pipeline(input_filename)
