#include <ros/ros.h>
#include <std_msgs/String.h>
#include <serial/serial.h>

// int main(int argc, char **argv)
// {
//     // Initialize the ROS system
//     ros::init(argc, argv, "ros_glove_controller");

//     // Establish this program as a ROS node
//     ros::NodeHandle nh;

//     // Send a log message
//     ROS_INFO("ROS Glove Controller Initialized");

//     // Spin to keep the node running
//     ros::spin();

//     return 0;
// }

serial::Serial ser;

void readGloveData(ros::Publisher &pub) {
    try {
        if (ser.available()) {
            std_msgs::String msg;
            msg.data = ser.read(ser.available());
            ROS_INFO("Glove Data: %s", msg.data.c_str());
            pub.publish(msg);
        }
    } catch (serial::IOException &e) {
        ROS_ERROR("Serial Exception: %s", e.what());
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "glove_node");
    ros::NodeHandle nh;

    ros::Publisher glove_pub = nh.advertise<std_msgs::String>("glove_data", 10);

    try {
        ser.setPort("/dev/ttyUSB0");  // Adjust based on glove interface
        ser.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(to);
        ser.open();
    } catch (serial::IOException &e) {
        ROS_ERROR("Unable to open port");
        return -1;
    }

    if (ser.isOpen()) {
        ROS_INFO("Serial port opened");
    } else {
        return -1;
    }

    ros::Rate loop_rate(10);  // 10Hz
    while (ros::ok()) {
        readGloveData(glove_pub);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
