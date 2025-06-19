#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import csv
import time
from std_msgs.msg import Header 
from dataglove.msg import SensorsData #DataPrint
from vmg30 import *
# from sensor_msgs.msg import JointState    ### firstly used that, realised it's not correct


class GloveNode:
    
    def __init__(self):
        # Read parameters
        self.port = rospy.get_param('/dataglove_params/serial/port')
        self.baudrate = rospy.get_param('/dataglove_params/serial/baudrate')
        self.pub_rate = rospy.get_param('/dataglove_params/serial/rate')
        self.vmg30 = VMG30()
        
        # Serial connection
        self.serial_conn = self.vmg30.open_device() #serial.Serial(self.port, self.baudrate, timeout=0.1) OR open_device() from vmg30
        
        # ROS Publisher
        self.pub = rospy.Publisher('glove_data', SensorsData, queue_size=1) #DataPrint
        
        # Sensor data
        self.print_data = SensorsData()

        # Buffer
        # self.batch_size = rospy.get_param('buffer_size')
        # self.input_dim = input_dim
        # self.buffer = torch.zeros((batch_size, input_dim))
        # self.index = 0
        # self.model = self.load_model()
        
       
    
    def read_data(self):
        try:
            self.vmg30.read_stream()
            if self.vmg30.is_new_packet_available():
                sensors = self.vmg30.sensors  # 23-element array
                sensors_map = {
                    'thumb_2': 0, 'thumb_1': 1,
                    'index_2': 2, 'index_1': 3,
                    'middle_2': 4, 'middle_1': 5,
                    'ring_2': 6, 'ring_1': 7,
                    'little_2': 8, 'little_1': 9,
                    'palm_arch': 10,
                    'thumb_crossover': 12,
                    'press_thumb': 14, 'press_index': 15,
                    'press_middle': 16, 'press_ring': 17, 'press_little': 18,
                    'abd_thumb': 19, 'abd_index': 20,
                    'abd_ring': 21, 'abd_little': 22
                }

                for field, idx in sensors_map.items():
                    setattr(self.print_data, field, int(sensors[idx]))
                # self.print_data.header.stamp = rospy.Time.now()
                # self.print_data.thumb_2 = sensors[0]
                # self.print_data.thumb_1 = sensors[1]
                # self.print_data.index_2 = sensors[2]
                # self.print_data.index_1 = sensors[3]
                # self.print_data.middle_2 = sensors[4]
                # self.print_data.middle_1 = sensors[5]
                # self.print_data.ring_2 = sensors[6]
                # self.print_data.ring_1 = sensors[7]
                # self.print_data.little_2 = sensors[8]
                # self.print_data.little_1 = sensors[9]
                # self.print_data.palm_arch = sensors[10]
                # self.print_data.thumb_crossover = sensors[12]
                # self.print_data.press_thumb = sensors[14]
                # self.print_data.press_index = sensors[15]
                # self.print_data.press_middle = sensors[16]
                # self.print_data.press_ring = sensors[17]
                # self.print_data.press_little = sensors[18]
                # self.print_data.abd_thumb = sensors[19]
                # self.print_data.abd_index = sensors[20]
                # self.print_data.abd_ring = sensors[21]
                # self.print_data.abd_little = sensors[22]
                self.vmg30.reset_new_packet()
                self.pub.publish(self.print_data)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")


    def run(self):
        rate = rospy.Rate(self.pub_rate)
        self.vmg30.send_start_packet(0x0)
        time.sleep(1) 
        self.vmg30.send_start_packet(0x1)
        time.sleep(1) 
        while not rospy.is_shutdown():
            self.read_data()
            rate.sleep()
        self.vmg30.close_device()

if __name__ == '__main__':
    try:
        rospy.init_node('node_raw_dataglove')
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

