#!/usr/bin/env python3
import rospy
import random
from std_msgs.msg import Header
from dataglove.msg import NewDataPrint

def fake_data_stream():
    rospy.init_node('fake_glove_pub', anonymous=True)
    pub = rospy.Publisher('glove_data', NewDataPrint, queue_size=1)
    rate_hz = rospy.get_param('/dataglove_params/serial/rate', 50)  # fallback to 50Hz
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        msg = NewDataPrint()
        msg.header.stamp = rospy.Time.now()

        # Simulated sensor values (replace with patterns if needed)
        msg.thumb_2 = random.randint(0, 255)
        msg.thumb_1 = random.randint(0, 255)
        msg.index_2 = random.randint(0, 255)
        msg.index_1 = random.randint(0, 255)
        msg.middle_2 = random.randint(0, 255)
        msg.middle_1 = random.randint(0, 255)
        msg.ring_2 = random.randint(0, 255)
        msg.ring_1 = random.randint(0, 255)
        msg.little_2 = random.randint(0, 255)
        msg.little_1 = random.randint(0, 255)
        msg.palm_arch = random.randint(0, 255)
        msg.thumb_crossover = random.randint(0, 255)
        msg.press_thumb = random.randint(0, 1)
        msg.press_index = random.randint(0, 1)
        msg.press_middle = random.randint(0, 1)
        msg.press_ring = random.randint(0, 1)
        msg.press_little = random.randint(0, 1)
        msg.abd_thumb = random.randint(0, 255)
        msg.abd_index = random.randint(0, 255)
        msg.abd_ring = random.randint(0, 255)
        msg.abd_little = random.randint(0, 255)

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        fake_data_stream()
    except rospy.ROSInterruptException:
        pass
