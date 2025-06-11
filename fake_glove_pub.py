#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
import random

def main():
    rospy.init_node('fake_glove_pub')
    pub = rospy.Publisher('/glove_data', JointState, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['thumb', 'index', 'middle', 'ring', 'pinky']
        msg.position = [random.uniform(0, 1) for _ in msg.name]
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    main()
