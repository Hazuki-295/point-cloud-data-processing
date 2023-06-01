import os
import sys

import open3d as o3d


def down_sample(filename):
    # Load the input point cloud file
    pcd = o3d.io.read_point_cloud(filename)

    # Function uniformly down-samples the point cloud, evenly select 1 point for every k points
    down_sampled_pcd = pcd.uniform_down_sample(every_k_points=10)

    # Save the down-sampled point cloud to an output file, which has the same extension as the input
    if not os.path.exists("data/output"):
        os.makedirs("data/output")
    base_name, extension = os.path.splitext(os.path.basename(filename))
    output_filename = base_name + " - downsampled" + extension
    filepath = os.path.join("data/output", output_filename)
    o3d.io.write_point_cloud(filepath, down_sampled_pcd, write_ascii=True)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        filenames.append("data/input/bunny.ply")

    for index, input_filename in enumerate(filenames):
        print(f"Downsampling file [{index + 1}]: {input_filename}")
        down_sample(input_filename)
