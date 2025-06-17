#!/usr/bin/env python3
import rospy
import sys
import rosbag_pandas
import torch
from dataglove.msg import DataPrint
from std_msgs.msg import Float32MultiArray, Int16MultiArray 


def set_start_param(param_name, delay_seconds):
    """Sets a ROS parameter after a specified delay to trigger another node."""
    def _set_param(event):
        rospy.set_param(param_name, True)
        # rospy.loginfo(f"[Printer] Set param {param_name} to True after {delay_seconds} seconds")

    rospy.Timer(rospy.Duration(delay_seconds), _set_param, oneshot=True)


class PrintNode:
    def __init__(self, record=False):

        self.record = record
        self.index = 0
        self.batch_size = rospy.get_param('/dataglove_params/buffer_size')
        self.buffer = torch.zeros((self.batch_size, 21), dtype=torch.int16)  # Assuming 21 features as per DataPrint
        self.sub = rospy.Subscriber('glove_data', DataPrint, self.callback)
        self.buffer_pub = rospy.Publisher('glove_buffer', Int16MultiArray, queue_size=1)


    def callback(self, msg):
        values = [getattr(msg, slot) for slot in msg.__slots__ if slot != 'header']
        ## PRINTS just to check the situation
        # rospy.loginfo("\nReceived state:", values)
        # Convert to tensor
        data = torch.tensor(values, dtype=torch.float32)
        # If buffer is not full, append the data
        if self.index < self.batch_size:
            self.buffer[self.index] = data
            self.index += 1
        else:
            # If buffer is full, shift the data (sliding window)
            # set_start_param("/dataglove_params/start_sron", 1)     # Starts after 1 second
            # set_start_param("/dataglove_params/start_mixron", 2)   # Starts 1 second after the previous
            # set_start_param("/dataglove_params/start_lsm", 3)      # Starts 1 second after the previous
            # Shift up: drop the oldest (first) row and append the new one
            self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
            self.buffer[-1] = data
            # After updating the buffer
            buffer_msg = Int16MultiArray(data=self.buffer.flatten().tolist())
            # rospy.loginfo(f"Buffer updated: {self.buffer.tolist()}")
            self.buffer_pub.publish(buffer_msg)

     

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    record = "record" in sys.argv

    rospy.init_node('node_printer')

    try:
        node = PrintNode(record=record)
        node.run()
    except rospy.ROSInterruptException:
        pass



###########################
# print of the msg
 
        # rospy.loginfo(f"thumb_2: {msg.thumb_2}")
        # rospy.loginfo(f"thumb_1: {msg.thumb_1}")
        # rospy.loginfo(f"index_2: {msg.index_2}")
        # rospy.loginfo(f"index_1: {msg.index_1}")
        # rospy.loginfo(f"middle_2: {msg.middle_2}")
        # rospy.loginfo(f"middle_1: {msg.middle_1}")
        # rospy.loginfo(f"ring_2: {msg.ring_2}")
        # rospy.loginfo(f"ring_1: {msg.ring_1}")
        # rospy.loginfo(f"little_2: {msg.little_2}")
        # rospy.loginfo(f"little_1: {msg.little_1}")
        # rospy.loginfo(f"palm_arch: {msg.palm_arch}")
        # rospy.loginfo(f"thumb_crossover: {msg.thumb_crossover}")
        # rospy.loginfo(f"press_thumb: {msg.press_thumb}")
        # rospy.loginfo(f"press_index: {msg.press_index}")
        # rospy.loginfo(f"press_middle: {msg.press_middle}")
        # rospy.loginfo(f"press_ring: {msg.press_ring}")
        # rospy.loginfo(f"press_little: {msg.press_little}")
        # rospy.loginfo(f"abd_thumb: {msg.abd_thumb}")
        # rospy.loginfo(f"abd_index: {msg.abd_index}")
        # rospy.loginfo(f"abd_ring: {msg.abd_ring}")
        # rospy.loginfo(f"abd_little: {msg.abd_little}")


   ## another callback: converting .bag files to .csv files
        # if self.record:
        #     try:
        #         rosbag_pandas.bag_to_csv(msg, 'glove_data.csv')
        #         rospy.loginfo("Data recorded to glove_data.csv")
        #     except ImportError:
        #         rospy.logerr("rosbag_pandas module not found. Install it to record data.")
        #     except Exception as e:
        #         rospy.logerr(f"Error recording data: {e}")
