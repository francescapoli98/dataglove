#!/usr/bin/env python3
import rospy
import sys
import rosbag_pandas
import torch
import os
from dataglove.msg import SensorsData,ClassificationData
from std_msgs.msg import Float32MultiArray, Int16MultiArray 


class PrintNode:
    def __init__(self, record=False):

        self.record = record
        self.index = 0
        self.batch_size = rospy.get_param('/dataglove_params/buffer_size')
        self.buffer = torch.zeros((self.batch_size, 21), dtype=torch.int16)  # 21 features (sensors)
        self.timestamp_buffer = []  # Store timestamps for each sample in buffer
        self.sub_rawdata = rospy.Subscriber('glove_data', SensorsData, self.callback)
        self.buffer_pub = rospy.Publisher('glove_buffer', ClassificationData, queue_size=1) 
        log_path = os.path.join(os.path.expanduser('~'), 'latency_log.txt')
        self.log_file = open(log_path, 'a')


    def callback(self, msg):
        msg_time = msg.header.stamp
        values = [getattr(msg, slot) for slot in msg.__slots__ if slot != 'header'] 
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
            log_line = f"{avg_latency_ms:.2f}\n"

            self.log_file.write(log_line)
            self.log_file.flush()  # Make sure it is written immediately
            
            # After updating the buffer
            buffer_msg = ClassificationData()
            buffer_msg.header.stamp = rospy.Time.now()
            buffer_msg.data = self.buffer.flatten().tolist()


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


