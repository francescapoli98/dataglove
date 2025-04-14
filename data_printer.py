#!/usr/bin/env python3
import rospy
import pandas as pd
import threading
import sys
import os
import time
from sensor_msgs.msg import JointState
from rospkg import RosPack


class PrintNode:
    def __init__(self, record=False):
        rospy.init_node('data_printer', anonymous=True)

        self.pub_rate = rospy.get_param('~rate', 50)
        self.lock = threading.Lock()
        self.record = record

        if self.record:
            # Get the absolute path to the 'dataglove' package
            rospack = RosPack()
            pkg_path = rospack.get_path('dataglove')
            logs_dir = os.path.join(pkg_path, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            # Use timestamped filename to avoid overwriting
            # Create timestamped log file in that directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.csv_file = os.path.join(logs_dir, f"borraccia_{timestamp}.csv")
            # Create the CSV file and write the header
            with open(self.csv_file, "w") as f:
                f.write("timestamp,joint_name,position,velocity,effort\n")

        self.sub = rospy.Subscriber('glove_joint_states', JointState, self.callback)

    def callback(self, msg):
        print("\nReceived Joint State:")
        rospy.loginfo(msg)

        if self.record:
            timestamp = rospy.get_time()
            rows = []

            for i, name in enumerate(msg.name):
                position = msg.position[i] if i < len(msg.position) else None
                velocity = msg.velocity[i] if i < len(msg.velocity) else None
                effort = msg.effort[i] if i < len(msg.effort) else None
                rows.append([timestamp, name, position, velocity, effort])

            df = pd.DataFrame(rows, columns=["timestamp", "joint_name", "position", "velocity", "effort"])
            df.to_csv(self.csv_file, mode="a", header=False, index=False)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    record = "record" in sys.argv  # Check if 'record' is passed as an argument
    try:
        node = PrintNode(record=record)
        node.run()
    except rospy.ROSInterruptException:
        pass
