#!/usr/bin/env python3

import rospy
import time

def set_start_param(param_name, delay):
    rospy.loginfo(f"Waiting {delay} seconds before starting {param_name}...")
    time.sleep(delay)
    rospy.set_param(param_name, True)
    rospy.loginfo(f"{param_name} set to True")

def main():
    rospy.init_node('node_controller')
    
    # Ensure the parameters start as False
    # rospy.set_param("/start_sron", False)
    # rospy.set_param("/start_mixron", False)
    # rospy.set_param("/start_lsm", False)

    # Set them to True with delays (in seconds)
    set_start_param("/start_sron", 2)     # Starts after 2 seconds
    set_start_param("/start_mixron", 4)   # Starts 2 seconds after the previous
    set_start_param("/start_lsm", 6)      # Starts 2 seconds after the previous

    # rospy.loginfo("All model nodes started in sequence.")
    rospy.spin()

if __name__ == '__main__':
    main()
