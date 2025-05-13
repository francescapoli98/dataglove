#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import csv
import time
from std_msgs.msg import Header 
from dataglove.msg import DataPrint
from vmg30 import *
# from sensor_msgs.msg import JointState    ### firstly used that, realised it's not correct


class GloveNode:
    
    def __init__(self):
        rospy.init_node('data_getter', anonymous=True)
        # Read parameters
        self.port = rospy.get_param('/my_prefix/port')
        self.baudrate = rospy.get_param('/my_prefix/baudrate')
        self.pub_rate = rospy.get_param('/my_prefix/rate')
        self.vmg30 = VMG30()
        
        # Serial connection
        self.serial_conn = self.vmg30.open_device() #serial.Serial(self.port, self.baudrate, timeout=0.1) OR open_device() from vmg30
        
        # ROS Publisher
        self.pub = rospy.Publisher('glove_data', DataPrint, queue_size=1)
        
        # Sensor data
        self.print_data = DataPrint()
        self.print_data.name = [
            'thumb_2', 'thumb_1', 'index_2', 'index_1',
            'middle_2', 'middle_1', 'ring_2', 'ring_1',
            'little_2', 'little_1', 'palm_arch',
            'nop', 'thumb_crossover','nop', #nop to be deleted
            'press_thumb', 'press_index', 'press_middle',
            'press_ring', 'press_little',
            'abd_thumb', 'abd_index', 'abd_ring', 'abd_little'    
        ]
        # self.print_data.value = [0.0] * len(self.print_data.name)
    
    def read_data(self):
        try:
            self.vmg30.read_stream() #data
            # rospy.loginfo(self.vmg30.is_new_packet_available())
            if self.vmg30.is_new_packet_available():
                self.print_data.value = self.vmg30.sensors
                self.vmg30.reset_new_packet()
            self.print_data.header.stamp = rospy.Time.now()
            # rospy.loginfo(self.print_data)
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
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
