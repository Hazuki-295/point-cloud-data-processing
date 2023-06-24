import os
import sys
from collections import Counter

import numpy as np
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

    # Mask by intensity to acquire low intensity points
    intensity_threshold = 50
    mask = np.where(filtered_pcd.point.intensity.numpy().flatten() <= intensity_threshold, True, False)

    # Extract low intensity points, then perform DBSCAN on them
    low_intensity_pcd = filtered_pcd.select_by_mask(mask)
    labels = low_intensity_pcd.cluster_dbscan(eps=0.10, min_points=5)
    labels = labels.numpy()

    n_clusters = labels.max() + 1
    print(f"DBSCAN clustering return {n_clusters} clusters.\n")

    # Clusters will be labeled in a way that cluster with the most points is labeled 1
    if n_clusters >= 2:
        counter = Counter(labels)

        # Remove noise that be labeled -1
        del counter[-1]

        # Sort by count
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        rank = [item[0] for item in sorted_items]

        # Convert mask to index
        index_of_mask = [i for i, boolean_val in enumerate(mask) if boolean_val]

        # Add cluster attribute to the input point cloud
        cluster = np.zeros(len(mask)).astype(np.int32)
        for i, val in enumerate(labels):
            if val != -1:
                cluster[index_of_mask[i]] = rank.index(val) + 1
        filtered_pcd.point.cluster = np.reshape(cluster, (len(cluster), 1))
    else:
        print("Warning: DBSCAN clustering return less than 2 clusters.")


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
