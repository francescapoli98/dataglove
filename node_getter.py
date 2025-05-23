#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import csv
import time
from std_msgs.msg import Header 
from dataglove.msg import DataPrint, NewDataPrint
from vmg30 import *
# from sensor_msgs.msg import JointState    ### firstly used that, realised it's not correct


class GloveNode:
    
    def __init__(self):
        rospy.init_node('node_getter', anonymous=True)
        # Read parameters
        self.port = rospy.get_param('/my_prefix/port')
        self.baudrate = rospy.get_param('/my_prefix/baudrate')
        self.pub_rate = rospy.get_param('/my_prefix/rate')
        self.vmg30 = VMG30()
        
        # Serial connection
        self.serial_conn = self.vmg30.open_device() #serial.Serial(self.port, self.baudrate, timeout=0.1) OR open_device() from vmg30
        
        # ROS Publisher
        self.pub = rospy.Publisher('glove_data', NewDataPrint, queue_size=1)
        
        # Sensor data
        self.print_data = NewDataPrint()
        
       
    
    def read_data(self):
        try:
            self.vmg30.read_stream()
            if self.vmg30.is_new_packet_available():
                sensors = self.vmg30.sensors  # 23-element array
                
                self.print_data.header.stamp = rospy.Time.now()
                self.print_data.thumb_2 = int(sensors[0])
                self.print_data.thumb_1 = int(sensors[1])
                self.print_data.index_2 = int(sensors[2])
                self.print_data.index_1 = int(sensors[3])
                self.print_data.middle_2 = int(sensors[4])
                self.print_data.middle_1 = int(sensors[5])
                self.print_data.ring_2 = int(sensors[6])
                self.print_data.ring_1 = int(sensors[7])
                self.print_data.little_2 = int(sensors[8])
                self.print_data.little_1 = int(sensors[9])
                self.print_data.palm_arch = int(sensors[10])
                # self.print_data.nop = 0  # or remove from msg
                self.print_data.thumb_crossover = int(sensors[12])
                self.print_data.press_thumb = int(sensors[14])
                self.print_data.press_index = int(sensors[15])
                self.print_data.press_middle = int(sensors[16])
                self.print_data.press_ring = int(sensors[17])
                self.print_data.press_little = int(sensors[18])
                self.print_data.abd_thumb = int(sensors[19])
                self.print_data.abd_index = int(sensors[20])
                self.print_data.abd_ring = int(sensors[21])
                self.print_data.abd_little = int(sensors[22])

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
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass



############################################################################### 
############################################################################### 
##  old code:
# 
# 
# 
 # self.print_data.name = [
        #     'thumb_2', 'thumb_1', 'index_2', 'index_1',
        #     'middle_2', 'middle_1', 'ring_2', 'ring_1',
        #     'little_2', 'little_1', 'palm_arch',
        #     'nop', 'thumb_crossover','nop', #nop to be deleted
        #     'press_thumb', 'press_index', 'press_middle',
        #     'press_ring', 'press_little',
        #     'abd_thumb', 'abd_index', 'abd_ring', 'abd_little'    
        # ]
        # self.print_data.value = [0.0] * len(self.print_data.name)

        # self.print_data = [0.0] * (len(self.print_data)-1) 
    
    # def read_data(self):
    #     try:
    #         self.vmg30.read_stream() #data
    #         # rospy.loginfo(self.vmg30.is_new_packet_available())
    #         if self.vmg30.is_new_packet_available():
    #             self.print_data.value = self.vmg30.sensors
    #             self.vmg30.reset_new_packet()
    #         self.print_data.header.stamp = rospy.Time.now()
    #         # rospy.loginfo(self.print_data)
    #         self.pub.publish(self.print_data)
    #     except serial.SerialException as e:
    #         rospy.logerr(f"Serial error: {e}")
############################################################################### 
