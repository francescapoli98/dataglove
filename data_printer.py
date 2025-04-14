#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import transforms3d.euler as euler
from sensor_msgs.msg import JointState


class PrintNode:
    def __init__(self):
        rospy.init_node('data_printer', anonymous=True)
        
        # Read parameters
        # self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        # self.baudrate = rospy.get_param('~baudrate', 230400)
        self.pub_rate = rospy.get_param('~rate', 50)
        
        # # Serial connection
        # self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
        
    #     # ROS Subscriber
        self.sub = rospy.Subscriber('glove_joint_states', JointState, self.callback)
        
    @property
    def packet_tick(self):
        with self.lock:
            return self.__packet_tick

#### FROM LIBRARY METHODS ####
    # def print_values(self):
    #     print("Package Tick:", self.__packet_tick)
    #     print("Thumb values:", self.__sensors[ThumbPh1R], self.__sensors[ThumbPh2R])
    #     print("Index values:", self.__sensors[IndexPh1R], self.__sensors[IndexPh2R])
    #     print("Middle values:", self.__sensors[MiddlePh1R], self.__sensors[MiddlePh2R])
    #     print("Ring values:", self.__sensors[RingPh1R], self.__sensors[RingPh2R])
    #     print("Little values:", self.__sensors[LittlePh1R], self.__sensors[LittlePh2R])

    #     rpy_wrist = self.quat2rpy(self.__quat_wrist)
    #     print(
    #         f"Wrist rpy: " f"{rpy_wrist[0]:.3f} {rpy_wrist[1]:.3f} {rpy_wrist[2]:.3f}"
    #     )
    #     rpy_hand = self.quat2rpy(self.__quat_hand)
    #     print(f"Hand rpy: " f"{rpy_hand[0]:.3f} {rpy_hand[1]:.3f} {rpy_hand[2]:.3f}")

    #     print(
    #         f"Wrist quat: "
    #         f"{self.__quat_wrist[0]:.3f} {self.__quat_wrist[1]:.3f} {self.__quat_wrist[2]:.3f} {self.__quat_wrist[3]:.3f}"
    #     )
    #     print(
    #         f"Hand quat: "
    #         f"{self.__quat_hand[0]:.3f} {self.__quat_hand[1]:.3f} {self.__quat_hand[2]:.3f} {self.__quat_hand[3]:.3f}"
    #     )        
    

    def callback(self, msg):
        print("\nReceived Joint State:")
        for name, pos in zip(msg.name, msg.position):
            print(f"{name}: {pos:.2f}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = PrintNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
