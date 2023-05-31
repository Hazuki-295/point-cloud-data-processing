import open3d as o3d
import os


def remove_outlier(input_file):
    pcd = o3d.io.read_point_cloud(input_file)

    # Estimate normals for the point cloud
    pcd.estimate_normals()

    # Remove the outlier points using statistical outlier removal function
    _, indexes = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # should use remove_radius_outlier

    inlier_cloud = pcd.select_by_index(indexes)
    outlier_cloud = pcd.select_by_index(indexes, invert=True)

    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # gray color for the reserved points
    outlier_cloud.paint_uniform_color([1, 0, 0])  # red color for the removed points

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # Save the down_sampled point cloud as a new .ply file
    if not os.path.exists("data"):
        os.makedirs("data")

    filename = os.path.basename(input_file)
    base_name, extension = os.path.splitext(filename)

    inlier_filename = base_name + " - removed - inlier" + ".pts"
    outlier_filename = base_name + " - removed - outlier" + ".pts"
    inlier_filepath = os.path.join("data", inlier_filename)
    outlier_filepath = os.path.join("data", outlier_filename)

    o3d.io.write_point_cloud(inlier_filepath, inlier_cloud, write_ascii=False)
    o3d.io.write_point_cloud(outlier_filepath, outlier_cloud, write_ascii=False)


if __name__ == '__main__':
    filename = "data/input/iScan-Pcd-1-1.pcd"
    remove_outlier(filename)
