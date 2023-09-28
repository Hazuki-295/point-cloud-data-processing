import os
import sys

import open3d as o3d


def preprocess(file_path):
    # Load the input point cloud file
    pcd = o3d.t.io.read_point_cloud(file_path)

    # Uniform down sample, evenly select 1 point for every k points
    down_sampled_pcd = pcd.uniform_down_sample(every_k_points=10)

    # Statistical outlier removal
    filtered_pcd, mask = down_sampled_pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

    # Save the preprocessed point cloud to an output file
    preprocessed_filename = os.path.join(output_path, base_name + " - preprocessed" + extension)
    o3d.t.io.write_point_cloud(preprocessed_filename, filtered_pcd)


if __name__ == "__main__":
    # File directory
    input_path = "data/input/"
    output_path = "data/preprocessed/"
    os.makedirs(output_path, exist_ok=True)

    # List of file paths to process
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        file_list = [os.path.join(input_path, f"iScan-Pcd-1-{i}.ply") for i in range(1, 6)]

    # Process each file
    print("Point cloud preprocessing...")
    for index, input_file_path in enumerate(file_list):
        # Prompt current file path
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))
        print(f"Input [{index + 1}]: {input_file_path}")
        preprocess(input_file_path)
