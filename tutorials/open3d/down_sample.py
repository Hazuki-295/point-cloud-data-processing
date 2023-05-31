import os
import open3d as o3d


def down_sample(filename):
    # Load the input point cloud file.
    pcd = o3d.io.read_point_cloud(filename)

    # Function uniformly down-samples the point cloud, evenly select 1 point for every k points.
    down_sampled_pcd = pcd.uniform_down_sample(every_k_points=10)

    # Save the down-sampled point cloud to an output file, which has the same extension as the input.
    if not os.path.exists("data/output"):
        os.makedirs("data/output")
    base_name, extension = os.path.splitext(os.path.basename(filename))
    output_filename = base_name + " - downsampled" + extension
    filepath = os.path.join("data/output", output_filename)
    o3d.io.write_point_cloud(filepath, down_sampled_pcd, write_ascii=True)


if __name__ == "__main__":
    input_filename = "data/input/bunny.ply"
    print("Down-sampling.")
    down_sample(input_filename)
    print("Complete.")
