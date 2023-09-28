import bisect
import json
import math
import os
import re
import sys

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata


def format_axes(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_formatter('{:.1f}'.format)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter('{:.1f}'.format)


# Slice the point cloud, then draw a depth image and a cross-section image on each slice
def visualization(file_path, every_k_meter=10):
    # Load the input point cloud file
    pcd = o3d.t.io.read_point_cloud(file_path)

    # Slice the point cloud every k meters
    number_of_slices = int((end_mileage - start_mileage) // every_k_meter)
    current_entry["number_of_slices"] = number_of_slices

    y_boundary = []
    y_split_pos = [start_mileage + i * every_k_meter for i in range(number_of_slices + 1)]
    for i in range(len(y_split_pos) - 1):
        boundary = [y_split_pos[i], y_split_pos[i + 1]]
        y_boundary.append(boundary)

    print(f"- Start mileage: {start_mileage:.2f}, end mileage: {end_mileage:.2f}")
    print(f"- Split the point cloud into {len(y_boundary)} slices: {y_boundary}")

    # Geometric constraint to crop track region
    x_min, x_max = [-4.0, 8.0]
    z_min, z_max = [-math.inf, 1.0]
    for index_boundary, (y_min, y_max) in enumerate(y_boundary):
        min_bound = np.array([x_min, y_min, z_min])
        max_bound = np.array([x_max, y_max, z_max])
        bounding_box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_slice = pcd.crop(bounding_box)

        # Slice number
        slice_number = index_boundary + 1
        slice_name = f"iScan-Pcd-1-{i_value} - slice {slice_number}"

        # Extract points
        coordinates = pcd_slice.point.positions.numpy()
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        intensity = pcd_slice.point.intensity.numpy().flatten()
        cluster = pcd_slice.point.cluster.numpy().flatten()

        point_left = coordinates[cluster == 1]
        point_right = coordinates[cluster == 2]

        # Create a figure and define the grid layout
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"Slice Profile — iScan-Pcd-1-{i_value}", fontsize=14)
        gs = fig.add_gridspec(8, 2)  # Define a grid for subplots

        # Create subplots
        ax_point = fig.add_subplot(gs[:4, 0], projection="3d")
        ax_cross = fig.add_subplot(gs[4:6, 0])
        ax_cross_prime = fig.add_subplot(gs[6:, 0])
        ax_depth = fig.add_subplot(gs[:7, 1])
        ax_text = fig.add_subplot(gs[7, 1])

        # 1. Point cloud (left-top, 3D subplot)
        # Axes setting
        ax_point.set(xlabel="x axis", ylabel="y axis", zlabel="z axis", zticks=[])
        ax_point.set_title(f"Transformed Point Cloud — Slice {slice_number}", fontsize=12)
        ax_point.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        ax_point.view_init(elev=45, azim=-115)

        # Plotting data
        ax_point.scatter3D(x, y, z, s=0.01, marker=',', c=intensity, cmap="gray")
        ax_point.plot([0.0, 0.0], [y_min, y_max], [1.0, 1.0], label="Centre line",
                      color='r', zorder=10)
        ax_point.plot(point_left[:, [0]], point_left[:, [1]], point_left[:, [2]], label="Left rail",
                      marker=',', markersize=10, color="cornflowerblue", zorder=10)
        ax_point.plot(point_right[:, [0]], point_right[:, [1]], point_right[:, [2]], label="Right rail",
                      marker=',', markersize=10, color="limegreen", zorder=10)
        ax_point.legend()

        # 2. Depth images (right, 2D subplot)
        # Axes setting
        ax_depth.set(xlabel="width (m)", ylabel="mileage (m)")
        ax_depth.set_title("Depth Image", fontsize=12)
        format_axes(ax_depth)

        # Define the grid on which to interpolate the points
        x_threshold = [x_min, x_max]
        y_threshold = [y_min, y_max]
        ax_depth.set_box_aspect(np.ptp(y_threshold) / np.ptp(x_threshold))  # Axis ratio is fixed

        # Interpolate the points onto the grid
        grid_x, grid_y = np.mgrid[x_threshold[0]:x_threshold[1]:1200j, y_threshold[0]:y_threshold[1]:1000j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="nearest")

        # Plot the interpolated grid
        z_threshold = [-1.0, 0.3]
        pc = ax_depth.pcolormesh(grid_x, grid_y, grid_z, vmin=z_threshold[0], vmax=z_threshold[1], cmap="RdBu_r")
        fig.colorbar(pc, ax=ax_depth)

        # 3. Cross-section image (left-bottom, 2D subplot)
        ideal_top_width = 3.1  # 3.1 m
        ideal_slope = 1 / 1.75  # 1:1.75
        sleeper_length = 2.60  # 2600 mm

        half_sleeper_length = sleeper_length / 2
        half_ideal_top_width = ideal_top_width / 2

        points = np.column_stack((x, z))
        x_split_pos = [-3.0, -1.8, -half_sleeper_length, half_sleeper_length, 1.8, 3.0]

        left_remainder = points[x <= x_split_pos[0]]
        left_area = points[[x_split_pos[0] < x < x_split_pos[2] for x in x]]
        sleeper_area = points[[x_split_pos[2] <= x <= x_split_pos[3] for x in x]]
        right_shoulder = points[[x_split_pos[3] < x < x_split_pos[4] for x in x]]
        right_slope = points[[x_split_pos[4] < x < x_split_pos[5] for x in x]]
        right_remainder = points[x_split_pos[5] <= x]

        # Linear fitting
        left_slope = left_area[left_area[:, 0] <= x_split_pos[1]]  # 1
        coefficients = np.polyfit(left_slope[:, 0], left_slope[:, 1], 1)
        slope, y_intercept = coefficients[0], coefficients[1]

        # Compute the left shoulder point first
        max_z_index = np.argmax(left_area[:, 1])
        max_point = left_area[max_z_index]
        x_split_pos[1] = max_point[0]

        # Then use interval endpoints to separate regions
        left_slope = left_area[left_area[:, 0] <= x_split_pos[1]]  # 2
        left_shoulder = left_area[left_area[:, 0] > x_split_pos[1]]

        # 3.1 Cross-section image (left-bottom 1)
        # Axes setting
        ax_cross.set(ylabel="Z")
        ax_cross.set_title("Cross-section Image", loc="left", fontsize=12)
        ax_cross.set_title("Inspection Profile")
        format_axes(ax_cross)

        # Plotting data
        regions = [sleeper_area, left_slope, left_shoulder, right_shoulder]
        labels = ["Sleeper region", "Left slope", "Left shoulder", "Right shoulder"]
        colors = ["red", "cornflowerblue", "orange", "limegreen"]
        for i in range(len(regions)):
            ax_cross.scatter(regions[i][:, 0], regions[i][:, 1], marker='^', s=0.01, c=colors[i], label=labels[i])

        point_on_top = [max_point[0], -half_sleeper_length, half_sleeper_length, x_split_pos[4]]
        colors = ["cornflowerblue", "red", "red", "limegreen"]
        for i in range(len(point_on_top)):
            ax_cross.plot([point_on_top[i], point_on_top[i]], [0, -1.0], linestyle='--', c=colors[i])

        remainder = np.vstack((left_remainder, right_slope, right_remainder))
        ax_cross.scatter(remainder[:, 0], remainder[:, 1], s=0.01, marker='^', c="gray")

        z_threshold = [-1.0, 1.0]  # different from depth image
        ax_cross.set_xlim(x_threshold)
        ax_cross.set_ylim(z_threshold)
        ax_cross.set_box_aspect(np.ptp(z_threshold) / np.ptp(x_threshold))
        ax_cross.set_yticks([-1.0, 0.0, 1.0])

        ax_cross.legend(markerscale=50)

        # 3.2 Cross-section image (left-bottom 2)
        # Axes setting
        ax_cross_prime.set(xlabel="X", ylabel="Z")
        ax_cross_prime.set_title("Comparison Diagram")
        format_axes(ax_cross_prime)

        # Plotting data
        ballast_bed_points = np.vstack((left_slope, left_shoulder, sleeper_area, right_shoulder, right_slope))

        # Plot ideal section and actual section
        ax_cross_prime.plot([-half_ideal_top_width, half_ideal_top_width], [0, 0], c="red", label="Ideal section")
        ax_cross_prime.scatter(ballast_bed_points[:, 0], ballast_bed_points[:, 1], s=0.01, marker='^', c="gray")

        width_offset = 2
        height_offset = ideal_slope * width_offset
        ax_cross_prime.plot([-half_ideal_top_width, -(half_ideal_top_width + width_offset)],  # ideal left slope
                            [0, -height_offset], c="red")
        ax_cross_prime.plot([half_ideal_top_width, half_ideal_top_width + width_offset],  # ideal right slope
                            [0, -height_offset], c="red")

        # Plot vertical lines
        point_on_top = [-half_sleeper_length, half_sleeper_length]
        for point in point_on_top:
            ax_cross_prime.plot([point, point], [0, -1.0], linestyle='--', c='r')
        ax_cross_prime.plot([0, 0], [0, -1.0], linestyle='--', c='k')

        # Left shoulder point
        index_point = bisect.bisect_left(ballast_bed_points[:, 0], max_point[0])
        max_point_prime = ballast_bed_points[index_point]

        ax_cross_prime.plot([max_point_prime[0]], [max_point_prime[1]], marker='o', c="cornflowerblue", zorder=20)
        ax_cross_prime.plot([max_point_prime[0], max_point_prime[0]], [max_point_prime[1], -1.0], linestyle='--',
                            c="cornflowerblue", zorder=20)
        ax_cross_prime.annotate("Left shoulder",
                                xy=(max_point_prime[0] - 0.05, max_point_prime[1] + 0.05),
                                xytext=(-2.8, 0.25),
                                arrowprops=dict(facecolor="cornflowerblue", headwidth=10, headlength=10))

        ax_cross_prime.legend(loc="lower right")

        x_threshold_prime = [-3.0, 3.0]
        z_threshold_prime = [-1.0, 0.5]
        ax_cross_prime.set_xlim(x_threshold_prime)
        ax_cross_prime.set_ylim(z_threshold_prime)
        ax_cross_prime.set_box_aspect(np.ptp(z_threshold_prime) / np.ptp(x_threshold_prime))
        ax_cross_prime.set_yticks([-1.0, -0.5, 0.0, 0.5])

        # 4. Statistic data
        left_slope_value = 1 / slope
        left_shoulder_width = (-max_point[0] - half_sleeper_length) * 1000  # mm

        text = f"Left slope — 1 : {left_slope_value:.2f}"
        text = text + "        " + f"Left shoulder width — {left_shoulder_width:.0f} mm"

        ax_text.text(0, 0, text, ha="left", va="center", fontsize=12)

        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)

        ax_text.axis("off")

        # Plotting complete, save the image
        plt.tight_layout()

        image_filename = os.path.join(output_path, slice_name + ".png")
        fig.savefig(image_filename, format="png", dpi=300)


if __name__ == "__main__":
    # File directory
    input_path = "data/transformed/"
    output_path = "data/output/"
    os.makedirs(output_path, exist_ok=True)

    # List of file paths to process
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
    else:
        file_list = [os.path.join(input_path, f"iScan-Pcd-1-{i} - transformed.ply") for i in range(1, 6)]

    json_file_path = os.path.join(input_path, "analysis_results.json")
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Process each file
    print("Analysis results visualization...")
    for index, input_file_path in enumerate(file_list):
        # Prompt current file path
        base_name, extension = os.path.splitext(os.path.basename(input_file_path))
        print(f"Input [{index + 1}]: {input_file_path}")

        pattern = r"iScan-Pcd-1-(\d+)"
        i_value = int(re.search(pattern, input_file_path).group(1))

        current_file = f"iScan-Pcd-1-{i_value}.ply"
        current_entry = next((entry for entry in json_data["files"] if entry["filename"] == current_file), None)

        start_mileage = current_entry["start_mileage"]
        end_mileage = current_entry["end_mileage"]

        visualization(input_file_path)

    # Save the updated JSON data to the output directory
    json_file_path = os.path.join(output_path, "analysis_results.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
