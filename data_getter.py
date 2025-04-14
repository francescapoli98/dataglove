#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import transforms3d.euler as euler
from sensor_msgs.msg import JointState


class GloveNode:
    def __init__(self):
        rospy.init_node('data_getter', anonymous=True)
        
        # Read parameters
        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        self.baudrate = rospy.get_param('~baudrate', 230400)
        self.pub_rate = rospy.get_param('~rate', 50)
        
        # Serial connection
        self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
        
        # ROS Publisher
        self.pub = rospy.Publisher('glove_joint_states', JointState, queue_size=10)
        
        # Sensor data
        self.joint_state = JointState()
        self.joint_state.name = [
            'thumb_1', 'thumb_2', 'index_1', 'index_2',
            'middle_1', 'middle_2', 'ring_1', 'ring_2',
            'little_1', 'little_2', 'palm_arch', 'thumb_crossover',
            'abd_thumb', 'abd_index', 'abd_ring', 'abd_little',
            'press_thumb', 'press_index', 'press_middle',
            'press_ring', 'press_little'
        ]
        self.joint_state.position = [0.0] * len(self.joint_state.name)
        
        self.lock = threading.Lock()
        # self.__sensors = [0.0] * 23  # replace 23 with actual sensor count

        
    def read_data(self):
        """Reads data from the glove and updates joint states."""
        try:
            data = self.serial_conn.read()  # Read enough bytes for a packet
            rospy.loginfo(f"Raw data: {data}")
            # if self.serial_conn:
            #     rospy.loginfo("Serial connection is open")
            if data:
                self.joint_state.position = self.decode_sensor_values(data)
                self.joint_state.header.stamp = rospy.Time.now()
                # self.pub.publish(self.joint_state)
                rospy.loginfo(self.joint_state)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")
        
    #### THIS I GOT FROM LIBRARY METHODS ####

    # @staticmethod
    # def quat2rpy(quat):
    #     """
    #     Compute roll pitch and yaw angles from quaternion values (radians)
    #     """
    #     return np.degrees(euler.quat2euler(quat, "sxyz"))

    # @staticmethod
    # def compute_rpy(quat):
    #     # original from cpp sources

    #     rpy = np.zeros(3, dtype=np.float32)

    #     ratio = 180.0 / math.pi
    #     rpy[0] = -ratio * math.atan2(
    #         2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
    #         1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
    #     )
    #     rpy[1] = -ratio * math.asin(2.0 * (quat[0] * quat[2] - quat[3] * quat[1]))
    #     rpy[2] = ratio * math.atan2(
    #         2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
    #         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
    #     )
    #     return rpy

    # @property
    # def rpy_wrist(self):
    #     """
    #     Get a copy of the wrist orientation in euler angles in degrees
    #     """
    #     with self.lock:
    #         return self.quat2rpy(self.__quat_wrist)

    # @property
    # def rpy_hand(self):
    #     """
    #     Get a copy of the hand orientation in euler angles in degrees
    #     """
    #     with self.lock:
    #         return self.quat2rpy(self.__quat_hand)

    # def reset_new_packet(self):
    #     """
    #     Reset new available packet flag
    #     """
    #     with self.lock:
    #         self.__new_packet_available = False

    # def is_new_packet_available(self):
    #     with self.lock:
    #         return self.__new_packet_available

    # @property
    # def packet_tick(self):
    #     with self.lock:
    #         return self.__packet_tick

    # # def decode_sensor_values(self, data): 
    # #     """Decodes raw data from the glove into sensor values."""
    # #     # Convert data
    # #     """
    # #     Get sensor value
    # #     :param index index of the requested finger
    # #     :return sensor measure
    # #     :raises ValueError
    # #     """

    # #     if 0 <= index < 23:
    # #         raise ValueError("No such sensor")

    # #     with self.lock:
    # #         return self.__sensors[index]
    # def decode_sensor_values(self, data):
    #     """
    #     Decodes raw data from the glove into sensor values.
    #     """
    #     with self.lock:
    #         # Example: return dummy sensor values from data, you’ll need to implement actual decoding
    #         return [float(i) for i in range(23)]


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
    
    def run(self):
        rate = rospy.Rate(self.pub_rate)
        while not rospy.is_shutdown():
            self.read_data()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
