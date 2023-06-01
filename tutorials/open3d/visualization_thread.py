import os
import sys
import threading

import open3d as o3d


def visualization(filename):
    # Load a point cloud file
    pcd = o3d.io.read_point_cloud(filename)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualization — {}".format(os.path.basename(filename)),
                      width=1000, height=800, left=200, top=150)
    vis.add_geometry(pcd)

    # Run the visualizer
    while True:
        vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    filenames = []

    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print("No input filenames passed.\n")
        filenames.append("data/input/bunny.ply")

    # Create a thread for each object
    threads = []
    for index, input_filename in enumerate(filenames):
        print(f"Visualize file [{index + 1}]: {input_filename}")
        thread = threading.Thread(target=visualization, args=(input_filename,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
