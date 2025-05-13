#!/usr/bin/env python3
import rospy
import pandas as pd
import threading
import sys
import os
import time
import csv
from dataglove.msg import DataPrint
from rospkg import RosPack
from vmg30 import *



class PrintNode:
    def __init__(self, record=False):
        rospy.init_node('data_printer', anonymous=True)

        self.record = record 

        self.sub = rospy.Subscriber('glove_data', DataPrint, self.callback)
        # self.sub = rospy.Subscriber('glove_joint_states', JointState, self.callback)

    def callback(self, msg):
        print("\nReceived state:")
        for i in msg.name, msg.value:
            print(i)

################# USA ROSBAG E POI CONVERTI .bag IN .csv ################################
      
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    record = "record" in sys.argv  # Check if 'record' is passed as an argument
    try:
        node = PrintNode(record=record)
        node.run()
    except rospy.ROSInterruptException:
        pass