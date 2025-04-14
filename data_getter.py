#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import transforms3d.euler as euler
from sensor_msgs.msg import JointState


# Packet Fields
PKT_TYPE_NONE = 0x0  # stop streaming
PKT_TYPE_QUAT_FINGER = 0x1  # start streaming quaternion and calibrated sensor values
PKT_TYPE_RAW_FINGER = 0x3  # start streaming raw values

PKT_HEADER = ord("$")
PKT_ENDCAR = ord("#")

PKT_CMD_UPDATE_VIBRO = 0x60
PKT_CMD_START_SAMPLING = 0x0A

class GloveNode:
    def __init__(self):
        rospy.init_node('data_getter', anonymous=True)
        
        # Read parameters
        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        self.baudrate = rospy.get_param('~baudrate', 230400)
        self.pub_rate = rospy.get_param('~rate', 1)
        
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
        # self.lock = threading.Lock()
        # self.__sensors = [0.0] * 23  # replace 23 with actual sensor count
    def send_start_packet(self, packet_type):
        """
        Send start streaming packet
        :param packet_type ->
            PKT_NONE = stop streaming
            PKT_QUAT_FINGER = start streaming quaternion and calibrated sensor values
            PKT_RAW_FINGER = start streaming raw values

        :return 0 if packet successfully sent, 1 if error
        """
        # |$|PKT_CMD_START|payload_length|packet_type|BCC|#|
        payload_length = 3
        buffer = bytearray(
            [
                PKT_HEADER,
                PKT_CMD_START_SAMPLING,
                payload_length,
                packet_type,
                0x00,
                PKT_ENDCAR,
            ]
        )

        buffer[4] = sum(buffer[:4]) & 0xFF  # BCC

        return self.send_packet(buffer)

    def send_packet(self, packet):
        """
        Send a packet

        :return 0 if packet successfully sent, 1 otherwise
        """
        if not self.serial_conn:
            return 1

        try:
            self.serial_conn.write(packet)
            print(f"Sent {len(packet)} bytes")

            return 0
        except Exception as e:
            print(f"Error Sending a packet: {e}")
            return 1
        
    def read_data(self):
        """Reads data from the glove and updates joint states."""
        try:
            data = self.serial_conn.read(50)  
            # rospy.loginfo(f"Raw data: {data}")
            # if self.serial_conn:
            #     rospy.loginfo("Serial connection is open")
            if data:
                self.joint_state.position = self.decode_sensor_values(data)
                self.joint_state.header.stamp = rospy.Time.now()
                self.pub.publish(self.joint_state)
                rospy.loginfo(self.joint_state)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")
        
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
    def decode_sensor_values(self, data):
        """
        Decodes raw data from the glove into sensor values.
        """
        # with self.lock:
            # Example: return dummy sensor values from data, you’ll need to implement actual decoding
        return [float(i) for i in data]
    
    def run(self):
        rate = rospy.Rate(self.pub_rate)
        self.send_start_packet(0x0)
        rate.sleep()
        self.send_start_packet(0x1)
        rate.sleep()
        while not rospy.is_shutdown():
            self.read_data()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = GloveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

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
