import os
import sys
from collections import Counter

import numpy as np
import open3d as o3d
from scipy import interpolate
from scipy.spatial import cKDTree


def curve_fitting(point):
    # Input should be points that stored in a (n, 3) ndarray
    point_num = point.shape[0]
    point = np.transpose(point)  # (n, 3) -> (3, n)

    # Find the B-spline representation of an 3-D curve
    tck, u = interpolate.splprep(point)
    u_prime = np.linspace(u.min(), u.max(), point_num * 2)
    knots = interpolate.splev(u_prime, tck)
    curve_point = np.column_stack((knots[0], knots[1], knots[2]))

    # Return fitted curve point in a (n, 3) ndarray
    return curve_point


def transform(filename):
    # Load the input point cloud file
    pcd = o3d.t.io.read_point_cloud(filename)

    # Part 1. DBSCAN clustering
    # Mask by intensity to acquire low intensity points
    intensity_threshold = 50
    mask = np.where(pcd.point.intensity.numpy().flatten() <= intensity_threshold, True, False)

    # Extract low intensity points, then perform DBSCAN on them
    low_intensity_pcd = pcd.select_by_mask(mask)
    labels = low_intensity_pcd.cluster_dbscan(eps=0.10, min_points=5)
    labels = labels.numpy()

    n_clusters = labels.max() + 1
    print(f"DBSCAN clustering return {n_clusters} clusters.\n")

    if n_clusters < 2:
        print("Warning: DBSCAN clustering return less than 2 clusters.")
        sys.exit(1)

    # Clusters will be labeled in a way that cluster with the most points is labeled 1
    counter = Counter(labels)
    del counter[-1]  # Remove noise that be labeled -1

    # Sort by count
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    rank = [item[0] for item in sorted_items]

    # Convert mask to index
    index_of_mask = [i for i, boolean_val in enumerate(mask) if boolean_val]

    # Add cluster attribute to the input point cloud
    cluster = np.zeros(len(mask)).astype(np.int32)
    for i, val in enumerate(labels):
        if val != -1:
            cluster[index_of_mask[i]] = rank.index(val) + 1
    pcd.point.cluster = np.reshape(cluster, (len(cluster), 1))

    # Part 2. Curve fitting
    # Step 1. Fit two curves on both left and right rails
    left_rail_mask = (cluster == 1)
    left_rail = pcd.select_by_mask(left_rail_mask)
    point_left = left_rail.point.positions.numpy()
    curve_point_left = curve_fitting(point_left)

    right_rail_mask = (cluster == 2)
    right_rail = pcd.select_by_mask(right_rail_mask)
    point_right = right_rail.point.positions.numpy()
    curve_point_right = curve_fitting(point_right)

    # Step 2. Calculate the centre line, then fit a curve on it
    # Build a kd-tree from the left curve points
    kdtree = cKDTree(curve_point_left)

    # Query the kd-tree to find the nearest neighbor and its distance
    distance, nearest_index = kdtree.query(curve_point_right)

    # Get the nearest point from the left curve points
    nearest_point = curve_point_left[nearest_index]

    # Points on the centre line
    point_centre = (nearest_point + curve_point_right) / 2

    # Fit a curve on centre line
    curve_point_centre = curve_fitting(point_centre)

    # Step 3. Store three fitted curves to a separate file (debug)
    point = np.vstack((curve_point_left, curve_point_right, curve_point_centre))

    colors = np.vstack((np.full((curve_point_left.shape[0], 3), [0.0, 0.0, 1.0]),  # color in blue
                        np.full((curve_point_right.shape[0], 3), [0.0, 1.0, 0.0]),  # color in green
                        np.full((curve_point_centre.shape[0], 3), [1.0, 0.0, 0.0])))  # color in red

    pcd_curve = o3d.t.geometry.PointCloud()
    pcd_curve.point.positions = point
    pcd_curve.point.colors = colors

    # Save the fitted curves for comparison
    base_name, extension = os.path.splitext(os.path.basename(filename))
    curve_filename = os.path.join("data/preprocessed/", base_name.replace("preprocessed", "curve") + extension)
    o3d.t.io.write_point_cloud(curve_filename, pcd_curve)

    # Part 3. RANSAC
    plane_model, _ = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model.numpy()  # ax + by + cz + d = 0

    # Part 4. Coordinate Transformation
    # Coordinates in original coordinate system
    coordinates = pcd.point.positions.numpy()

    # Prepare for coordinate calculation
    normal_z = np.array([a, b, c])
    if np.dot(normal_z, curve_point_centre[0]) + d < 0:  # centre curve points should have z > 0
        normal_z = -normal_z
    magnitude_z = np.linalg.norm(normal_z)

    normal_y = np.diff(curve_point_centre, axis=0)
    normal_y = np.insert(normal_y, 0, [0., 0., 0.], axis=0)
    distances = np.linalg.norm(normal_y, axis=1)
    cumulate_y = np.cumsum(distances)

    kdtree_centre = cKDTree(curve_point_centre)
    distance, nearest_index = kdtree_centre.query(coordinates)
    nearest_point = curve_point_centre[nearest_index]

    PQ = coordinates - nearest_point
    magnitude_PQ = distance
    cross_product = np.cross(normal_z, PQ)
    sin_theta = np.linalg.norm(cross_product, axis=1) / (magnitude_z * magnitude_PQ)

    dot_product = np.sum(cross_product * normal_y[nearest_index], axis=1)
    sign_x = np.where(dot_product > 0, 1, -1)

    # Coordinate calculation
    x = sign_x * magnitude_PQ * sin_theta
    y = cumulate_y[nearest_index]
    z = (np.dot(coordinates, normal_z) + d) / magnitude_z

    pcd.point.positions = np.column_stack((x, y, z))
    output_filename = os.path.join(output_path, base_name.replace("preprocessed", "transformed") + extension)
    o3d.t.io.write_point_cloud(output_filename, pcd)


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("Warning: No input filenames passed.")
        exit(-1)

    output_path = "data/transformed"
    os.makedirs(output_path, exist_ok=True)

    for index, input_filename in enumerate(filenames):
        print(f"Point cloud preprocessing [{index + 1}]: {input_filename}")
        transform(input_filename)