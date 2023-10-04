import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial import cKDTree


def curve_fitting(point, point_num=None):
    # Remove duplicate points
    point = np.unique(point, axis=0)

    # Input should be points that stored in a (n, 3) ndarray
    if point_num is None:
        point_num = point.shape[0]

    # Find the B-spline representation of an 3-D curve
    tck, u = interpolate.splprep(point.T)  # (n, 3) -> (3, n), B-spline representation
    u_prime = np.linspace(u.min(), u.max(), point_num)
    knots = interpolate.splev(u_prime, tck)

    # Return fitted curve point in a (n, 3) ndarray
    curve_point = np.array(knots).T
    return curve_point


# If transform raise an exception, check the DBSCAN clustering results
def dbscan_debug(coordinates, cluster, sorted_items, debug=False):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"DBSCAN clustering results — iScan-Pcd-1-{i_value}", fontsize=14)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.set_title(f"Five largest clusters", fontsize=12)

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_title("Left and Right Rails", fontsize=12)

    ax3 = fig.add_subplot(gs[1, 1], projection='3d')
    ax3.set_title("other clusters", fontsize=12)

    for ax, indices in [[ax1, list(range(0, 5))], [ax2, list(range(0, 2))], [ax3, list(range(2, 5))]]:
        for i in indices:
            var, count = sorted_items[i]
            point_cluster = coordinates[cluster == i + 1]
            x, y, z = point_cluster[:, 0], point_cluster[:, 1], point_cluster[:, 2]
            ax.scatter3D(x, y, z, label=f"cluster {i + 1}: {count}")
        ax.legend(loc="upper right")

    plt.tight_layout()

    if debug:
        plt.show()
    else:
        image_filename = os.path.join(dbscan_image_path, f"iScan-Pcd-1-{i_value}.png")
        fig.savefig(image_filename, format="png", dpi=300)

    plt.close(fig)


def transform(file_path, every_k_meter=10, debug=False):
    # Whether to skip transformation
    if input_file_path in file_path_exclude:
        excluded_file = True
        print(f"- Current input file path has been excluded, skip transformation.")
    else:
        excluded_file = False

    # Check if there is an entry corresponding to the previous file
    if i_value > 1 and not excluded_file:
        previous_file = file_path_all_filtered[file_path_all_filtered.index(file_path) - 1]
        previous_i_value = int(re.search(pattern, previous_file).group(1))
        previous_filename = f"iScan-Pcd-1-{previous_i_value}.ply"
        previous_entry = next((entry for entry in json_data["files"] if entry["filename"] == previous_filename), None)
        if previous_entry is None:
            print(f"Error: Entry for previous file {previous_file} not found in JSON data.")
            exit(1)
        previous_end_mileage = previous_entry["end_mileage"]

        if previous_file == file_path_all[file_path_all.index(file_path) - 1]:
            previous_remainder_filename = f"iScan-Pcd-1-{previous_i_value} - remainder.ply"
            previous_remainder_filepath = os.path.join(preprocessed_path, previous_remainder_filename)
            remainder_pcd = o3d.t.io.read_point_cloud(previous_remainder_filepath)
            current_pcd = o3d.t.io.read_point_cloud(file_path)

            positions = np.vstack((remainder_pcd.point.positions.numpy(), current_pcd.point.positions.numpy()))
            intensity = np.vstack((remainder_pcd.point.intensity.numpy(), current_pcd.point.intensity.numpy()))

            pcd = o3d.t.geometry.PointCloud()
            pcd.point.positions = positions
            pcd.point.intensity = intensity
        else:
            pcd = o3d.t.io.read_point_cloud(file_path)
    else:
        previous_end_mileage = 0.0  # First file, or the file has been excluded
        pcd = o3d.t.io.read_point_cloud(file_path)

    # Part 1. DBSCAN clustering
    # Mask by intensity to acquire low intensity points
    intensity_threshold = 50
    mask = np.where(pcd.point.intensity.numpy().flatten() <= intensity_threshold, True, False)

    # Extract low intensity points, then perform DBSCAN on them
    low_intensity_pcd = pcd.select_by_mask(mask)
    labels = low_intensity_pcd.cluster_dbscan(eps=0.10, min_points=5)
    labels = labels.numpy()

    n_clusters = labels.max() + 1
    print(f"- DBSCAN clustering return {n_clusters} clusters.")

    if n_clusters < 2:
        print("- Error: DBSCAN clustering return less than 2 clusters.")
        sys.exit(1)

    # Clusters will be labeled in a way that cluster with the most points is labeled 1
    counter = Counter(labels)
    del counter[-1]  # Remove noise that be labeled -1

    # Sort by count
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    rank = [item[0] for item in sorted_items]
    print("- Five largest clusters:", sorted_items[:5])

    # Convert mask to index
    index_of_mask = [i for i, boolean_val in enumerate(mask) if boolean_val]

    # Add cluster attribute to the input point cloud
    cluster = np.zeros(len(mask)).astype(np.int32)
    for i, val in enumerate(labels):
        if val != -1:
            cluster[index_of_mask[i]] = rank.index(val) + 1
    pcd.point.cluster = np.reshape(cluster, (len(cluster), 1))

    # Coordinates in original coordinate system
    coordinates = pcd.point.positions.numpy()

    # Save the DBSCAN clustering results for debug
    dbscan_debug(coordinates, cluster, sorted_items, debug)

    # Skip transformation
    if excluded_file:
        return

    # Part 2. Curve fitting
    # Step 1. Fit two curves on both left and right rails
    point_left = coordinates[cluster == 1]
    point_right = coordinates[cluster == 2]
    try:
        curve_point_left = curve_fitting(point_left)
        curve_point_right = curve_fitting(point_right)
    except Exception as e:
        exception_name = type(e).__name__
        print(f"- Error: Caught an exception, {exception_name}: {e}")
        traceback.print_exc()
        dbscan_debug(pcd, cluster, sorted_items, debug=True)
        exit(1)

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
    curve_filename = os.path.join(preprocessed_path, base_name.replace("preprocessed", "curve") + extension)
    o3d.t.io.write_point_cloud(curve_filename, pcd_curve)

    # Part 3. RANSAC
    plane_model, _ = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model.numpy()  # ax + by + cz + d = 0

    # Part 4. Coordinate Transformation
    start_time = time.time()

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
    start_mileage = previous_end_mileage // every_k_meter * every_k_meter

    x = sign_x * magnitude_PQ * sin_theta
    y = start_mileage + cumulate_y[nearest_index]
    z = (np.dot(coordinates, normal_z) + d) / magnitude_z

    end_mileage = start_mileage + cumulate_y[-1]

    update_json(f"iScan-Pcd-1-{i_value}.ply", start_mileage, end_mileage)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"- Elapsed time: {elapsed_time:.2f} seconds.")

    # Crop the point cloud based on y coordinate
    mask = np.where(y >= (end_mileage // every_k_meter * every_k_meter), True, False)
    remainder_pcd = pcd.select_by_mask(mask)
    remainder_filename = os.path.join(preprocessed_path, base_name.replace("preprocessed", "remainder") + extension)
    o3d.t.io.write_point_cloud(remainder_filename, remainder_pcd)

    pcd.point.positions = np.column_stack((x, y, z))

    transformed_filename = os.path.join(transformed_path, base_name.replace("preprocessed", "transformed") + extension)
    o3d.t.io.write_point_cloud(transformed_filename, pcd)


# If JSON file doesn't exist, create an empty structure
def initialize_json(file_path):
    if not os.path.exists(file_path):
        template = {
            "resultName": "Railway Track Information",
            "lastModified": None,
            "files": []
        }
        with open(json_file_path, 'w') as file:
            json.dump(template, file, indent=4)


# Function to update the JSON structure with processing information
def update_json(filename, start_mileage, end_mileage):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "filename": filename,
        "start_mileage": start_mileage,
        "end_mileage": end_mileage,
        "length_of_slice": None,
        "number_of_slices": None,
        "slices": [],
        "lastModified": now
    }

    # Check if an entry with the same filename already exists
    existing_entry = next((entry for entry in json_data["files"] if entry["filename"] == filename), None)
    if existing_entry is None:
        json_data["files"].append(new_entry)
    else:
        existing_entry.update(new_entry)

    json_data["lastModified"] = now


if __name__ == "__main__":
    # File directory
    preprocessed_path = "data/preprocessed/"
    transformed_path = "data/transformed/"
    os.makedirs(transformed_path, exist_ok=True)

    dbscan_image_path = "data/transformed/dbscan_img"
    os.makedirs(dbscan_image_path, exist_ok=True)

    # List of file paths to process
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        file_list = [os.path.join(preprocessed_path, f"iScan-Pcd-1-{i} - preprocessed.ply") for i in range(1, 6)]

    # Files that should be skipped
    i_exclude = list(range(11, 17)) + list(range(28, 33)) + list(range(38, 49))
    file_path_exclude = [os.path.join(preprocessed_path, f"iScan-Pcd-1-{i} - preprocessed.ply") for i in i_exclude]

    file_path_all = [os.path.join(preprocessed_path, f"iScan-Pcd-1-{i} - preprocessed.ply") for i in range(1, 49)]
    file_path_all_filtered = []
    for item in file_path_all:
        if item not in file_path_exclude:
            file_path_all_filtered.append(item)

    json_file_path = os.path.join("data/", "analysis_results.json")
    initialize_json(json_file_path)
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Process each file
    print("Point cloud transforming...")
    for index, input_file_path in enumerate(file_list):
        # Prompt current file path
        print(f"Input [{index + 1}]: {input_file_path}")

        pattern = r"iScan-Pcd-1-(\d+)"
        i_value = int(re.search(pattern, input_file_path).group(1))
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))

        try:
            transform(input_file_path)
            print()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Program interrupted by user.")
            break

    # Save the updated JSON data to the output directory
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
