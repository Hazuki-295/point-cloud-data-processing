#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    /* Load the pcd file. */
    std::string filename = "../../data/input/iScan-Pcd-1-1 - preprocess.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        PCL_ERROR ("Couldn't read pcd file.\n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " + filename << std::endl;
    
    return 0;
}