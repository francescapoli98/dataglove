#include <ros/ros.h>
#include <std_msgs/String.h>

int main(int argc, char **argv)
{
    // Initialize the ROS system
    ros::init(argc, argv, "ros_glove_controller");

    // Establish this program as a ROS node
    ros::NodeHandle nh;

    // Send a log message
    ROS_INFO("ROS Glove Controller Initialized");

    // Spin to keep the node running
    ros::spin();

    return 0;
}