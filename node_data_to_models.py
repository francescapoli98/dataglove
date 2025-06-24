#!/usr/bin/env python3
import rospy
import sys
import rosbag_pandas
import torch
import os
from dataglove.msg import SensorsData,ClassificationData
from std_msgs.msg import Float32MultiArray, Int16MultiArray 


# def set_start_param(param_name, delay_seconds):
#     """Sets a ROS parameter after a specified delay to trigger another node."""
#     def _set_param(event):
#         rospy.set_param(param_name, True)
#         # rospy.loginfo(f"[Printer] Set param {param_name} to True after {delay_seconds} seconds")

#     rospy.Timer(rospy.Duration(delay_seconds), _set_param, oneshot=True)


class PrintNode:
    def __init__(self, record=False):

        self.record = record
        self.index = 0
        self.batch_size = rospy.get_param('/dataglove_params/buffer_size')
        self.buffer = torch.zeros((self.batch_size, 21), dtype=torch.int16)  # Assuming 21 features 
        self.timestamp_buffer = []  # Store timestamps for each sample in buffer
        self.sub_rawdata = rospy.Subscriber('glove_data', SensorsData, self.callback)
        self.buffer_pub = rospy.Publisher('glove_buffer', ClassificationData, queue_size=1) #Int16MultiArray
        log_path = os.path.join(os.path.expanduser('~'), 'latency_log.txt')
        self.log_file = open(log_path, 'a')
        # rospy.loginfo(f"[Printer] Latency log file opened")


    def callback(self, msg):
        msg_time = msg.header.stamp
        values = [getattr(msg, slot) for slot in msg.__slots__ if slot != 'header'] 
        ## PRINTS just to check the situation
        # rospy.loginfo("\nReceived state:", values)
        # Convert to tensor
        data = torch.tensor(values, dtype=torch.float32)
        
        # If buffer is not full, append the data
        if self.index < self.batch_size:
            self.buffer[self.index] = data
            self.timestamp_buffer.append(msg_time)
            self.index += 1
        else:
            # Buffer is full, update the buffer
            # Shift up: drop the oldest (first) row and append the new one
            self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
            self.buffer[-1] = data
            # Maintain timestamp buffer size same as batch_size
            self.timestamp_buffer.append(msg_time)
            if len(self.timestamp_buffer) > self.batch_size:
                self.timestamp_buffer.pop(0)

            # Compute average latency for the buffer batch
            now = rospy.Time.now()
            latencies = [(now - ts).to_sec() for ts in self.timestamp_buffer]
            avg_latency_ms = sum(latencies) / len(latencies) * 1000
            # rospy.loginfo(f"[Printer] Average latency for buffer batch: {avg_latency_ms:.2f} ms")
            log_line = f"{avg_latency_ms:.2f}\n"

            self.log_file.write(log_line)
            self.log_file.flush()  # Make sure it is written immediately
            
            # After updating the buffer
            # buffer_msg = Int16MultiArray(data=self.buffer.flatten().tolist())
            buffer_msg = ClassificationData()
            buffer_msg.header.stamp = rospy.Time.now()
            buffer_msg.data = self.buffer.flatten().tolist()


            # rospy.loginfo(f"Buffer updated: {self.buffer.tolist()}")
            self.buffer_pub.publish(buffer_msg)

     

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    record = "record" in sys.argv

    rospy.init_node('node_data_to_models')

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
