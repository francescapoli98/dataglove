#!/usr/bin/env python3
import rospy
import sys
from dataglove.msg import NewDataPrint

class PrintNode:
    def __init__(self, record=False):
        rospy.init_node('node_printer', anonymous=True)
        self.record = record
        self.sub = rospy.Subscriber('glove_data', NewDataPrint, self.callback)

    def callback(self, msg):
        rospy.loginfo("\nReceived state:")
        rospy.loginfo(f"thumb_2: {msg.thumb_2}")
        rospy.loginfo(f"thumb_1: {msg.thumb_1}")
        rospy.loginfo(f"index_2: {msg.index_2}")
        rospy.loginfo(f"index_1: {msg.index_1}")
        rospy.loginfo(f"middle_2: {msg.middle_2}")
        rospy.loginfo(f"middle_1: {msg.middle_1}")
        rospy.loginfo(f"ring_2: {msg.ring_2}")
        rospy.loginfo(f"ring_1: {msg.ring_1}")
        rospy.loginfo(f"little_2: {msg.little_2}")
        rospy.loginfo(f"little_1: {msg.little_1}")
        rospy.loginfo(f"palm_arch: {msg.palm_arch}")
        rospy.loginfo(f"thumb_crossover: {msg.thumb_crossover}")
        rospy.loginfo(f"press_thumb: {msg.press_thumb}")
        rospy.loginfo(f"press_index: {msg.press_index}")
        rospy.loginfo(f"press_middle: {msg.press_middle}")
        rospy.loginfo(f"press_ring: {msg.press_ring}")
        rospy.loginfo(f"press_little: {msg.press_little}")
        rospy.loginfo(f"abd_thumb: {msg.abd_thumb}")
        rospy.loginfo(f"abd_index: {msg.abd_index}")
        rospy.loginfo(f"abd_ring: {msg.abd_ring}")
        rospy.loginfo(f"abd_little: {msg.abd_little}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    record = "record" in sys.argv  
    try:
        node = PrintNode(record=record)
        node.run()
    except rospy.ROSInterruptException:
        pass
