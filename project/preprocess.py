import os
import sys

import open3d as o3d


def preprocess(filename):
    # Load the input point cloud file
    pcd = o3d.t.io.read_point_cloud(filename)

    # Uniform down sample, evenly select 1 point for every k points
    down_sampled_pcd = pcd.uniform_down_sample(every_k_points=10)

    # Statistical outlier removal
    filtered_pcd, mask = down_sampled_pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

    # Save the preprocessed point cloud to an output file
    base_name, extension = os.path.splitext(os.path.basename(filename))
    output_filename = os.path.join(output_path, base_name + " - preprocessed" + extension)
    o3d.t.io.write_point_cloud(output_filename, filtered_pcd)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        exit(-1)

    output_path = "data/preprocessed/"
    os.makedirs(output_path, exist_ok=True)

    for index, input_filename in enumerate(filenames):
        print(f"Point cloud preprocessing [{index + 1}]: {input_filename}")
        preprocess(input_filename)
