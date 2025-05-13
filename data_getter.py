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
        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        self.baudrate = rospy.get_param('~baudrate', 230400)
        self.pub_rate = rospy.get_param('~rate', 1)
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
        self.print_data.value = [0.0] * len(self.print_data.name)
    
    def read_data(self):
        try:
            self.print_data.value = self.vmg30.read_stream() #data
            
            self.print_data.header.stamp = rospy.Time.now()
            rospy.loginfo(self.print_data)
            self.pub.publish(self.print_data)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")
        
    def run(self):
        rate = rospy.Rate(self.pub_rate)
        self.vmg30.send_start_packet(0x0)
        time.sleep(1) #rate.sleep()
        self.vmg30.send_start_packet(0x1)
        time.sleep(1) #rate.sleep()
        while not rospy.is_shutdown():
            self.read_data()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass















##################################################################################################
##################################################################################################
##################################################################################################
    # @property
    # def sensors(self):
    #     """
    #     Get a copy of the sensors array
    #     """
    #     with self.lock:
    #         return np.copy(self.__sensors)

    # @property
    # def quaternion_wrist(self):
    #     """
    #     Get a copy of the wrist orientation in quaternion
    #     """
    #     with self.lock:
    #         return np.copy(self.__quat_wrist)

    # @property
    # def quaternion_hand(self):
    #     """
    #     Get a copy of the hand orientation in quaternion
    #     """
    #     with self.lock:
    #         return np.copy(self.__quat_hand)

    # # def __print_values(self):
    # #     print("Package Tick:", self.__packet_tick)
    # #     print("Thumb values:", self.__sensors[ThumbPh1R], self.__sensors[ThumbPh2R])
    # #     print("Index values:", self.__sensors[IndexPh1R], self.__sensors[IndexPh2R])
    # #     print("Middle values:", self.__sensors[MiddlePh1R], self.__sensors[MiddlePh2R])
    # #     print("Ring values:", self.__sensors[RingPh1R], self.__sensors[RingPh2R])
    # #     print("Little values:", self.__sensors[LittlePh1R], self.__sensors[LittlePh2R])

    # #     rpy_wrist = self.quat2rpy(self.__quat_wrist)
    # #     print(
    # #         f"Wrist rpy: " f"{rpy_wrist[0]:.3f} {rpy_wrist[1]:.3f} {rpy_wrist[2]:.3f}"
    # #     )
    # #     rpy_hand = self.quat2rpy(self.__quat_hand)
    # #     print(f"Hand rpy: " f"{rpy_hand[0]:.3f} {rpy_hand[1]:.3f} {rpy_hand[2]:.3f}")

    # #     print(
    # #         f"Wrist quat: "
    # #         f"{self.__quat_wrist[0]:.3f} {self.__quat_wrist[1]:.3f} {self.__quat_wrist[2]:.3f} {self.__quat_wrist[3]:.3f}"
    # #     )
    # #     print(
    # #         f"Hand quat: "
    # #         f"{self.__quat_hand[0]:.3f} {self.__quat_hand[1]:.3f} {self.__quat_hand[2]:.3f} {self.__quat_hand[3]:.3f}"
    # #     )        
