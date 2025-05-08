#!/usr/bin/env python3
import rospy
import pandas as pd
import threading
import sys
import os
import time
import csv
# from sensor_msgs.msg import JointState
from dataglove.msg import DataPrint
from rospkg import RosPack


class PrintNode:
    def __init__(self, record=False):
        rospy.init_node('data_printer', anonymous=True)

        self.pub_rate = rospy.get_param('~rate', 1)
        self.lock = threading.Lock()
        self.record = record

        if self.record:
            # Get the absolute path to the 'dataglove' package
            rospack = RosPack()
            pkg_path = rospack.get_path('dataglove')
            logs_dir = os.path.join(pkg_path, 'logs/test_logs')
            os.makedirs(logs_dir, exist_ok=True)
            # Use timestamped filename to avoid overwriting
            # Create timestamped log file in that directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.csv_file = os.path.join(logs_dir, f"test_{timestamp}.csv")
            # Create the CSV file and write the header
            # with open(self.csv_file, "w") as f:
            #     f.write("timestamp,name,value\n")#,velocity,effort\n")

        self.sub = rospy.Subscriber('glove_data', DataPrint, self.callback)
        # self.sub = rospy.Subscriber('glove_joint_states', JointState, self.callback)

    def callback(self, msg):
        print("\nReceived state:")
        # rospy.loginfo()

        if self.record:
            timestamp = rospy.get_time()
            rows = []

            for i, name in enumerate(msg.name):
                # rospy.loginfo(i)
                # rospy.loginfo(name)


                value = msg.value[i] if i < len(msg.value) else None
                # rospy.loginfo(value)
                # velocity = msg.velocity[i] if i < len(msg.velocity) else None
                # effort = msg.effort[i] if i < len(msg.effort) else None
                rows.append([timestamp, name, value])#, velocity, effort])

            df = pd.DataFrame(rows, columns=["timestamp", "name", "value"])
            # rospy.loginfo(df)
            # df = pd.DataFrame(rows, columns=["timestamp", "joint_name", "position", "velocity", "effort"])
            df.to_csv(self.csv_file, mode="a", header=True, index=False)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    record = "record" in sys.argv  # Check if 'record' is passed as an argument
    try:
        node = PrintNode(record=record)
        node.run()
    except rospy.ROSInterruptException:
        pass
    ### if I want to run the neural network on the csv file immediatly
    # finally:
    #     if record:
    #         from classification import MyNeuralNetwork  # Replace with your actual module/class
    #         nn = MyNeuralNetwork()
    #         nn.run_on_file(node.csv_file)
