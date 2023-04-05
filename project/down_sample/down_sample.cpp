#include <iostream>
#include <string>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/uniform_sampling.h>

int main() {
    /* Load the point cloud. */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string filename = "../../data/input/iScan-Pcd-1-1 - preprocess.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        PCL_ERROR ("Couldn't read pcd file.\n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->size() << " data points from " + filename << '\n' << std::endl;

    /* Down sample the point cloud. */
    float leaf_size = 0.01f;
    pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
    uniform_sampling.setInputCloud(cloud);
    uniform_sampling.setRadiusSearch(leaf_size);

    std::cout << "Down sampling..." << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    uniform_sampling.filter(*sampled_cloud);

    std::cout << "Complete.\n" << std::endl;

    /* Store down sampled point cloud. */
    std::string basename = std::filesystem::path(filename).stem().string();
    std::string output_filename = "../../data/output/" + basename + " - down_sampled.pcd";
    pcl::io::savePCDFileASCII(output_filename, *sampled_cloud);

    std::cout << "Sampled " << sampled_cloud->size() << " data points has stored to " + output_filename << std::endl;

    return 0;
}