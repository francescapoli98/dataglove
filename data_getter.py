#!/usr/bin/env python3
import rospy
import serial
import numpy as np
import math
import threading
import csv
from std_msgs.msg import Header 
from dataglove.msg import DataPrint
# import transforms3d.euler as euler        #### I don't think we need this
# from sensor_msgs.msg import JointState    ### firstly used that, realised it's not correct


# Packet Fields
PKT_TYPE_NONE = 0x0  # stop streaming
PKT_TYPE_QUAT_FINGER = 0x1  # start streaming quaternion and calibrated sensor values
PKT_TYPE_RAW_FINGER = 0x3  # start streaming raw values

PKT_HEADER = ord("$")
PKT_ENDCAR = ord("#")

PKT_CMD_UPDATE_VIBRO = 0x60
PKT_CMD_START_SAMPLING = 0x0A

class GloveNode:
    """
    GloveNode is a ROS node that interfaces with a data glove to read sensor data and publish it to a ROS topic.
    Attributes:
        port (str): The serial port to which the glove is connected. Default is '/dev/ttyUSB0'.
        baudrate (int): The baud rate for the serial connection. Default is 230400.
        pub_rate (int): The rate (in Hz) at which data is published. Default is 50.
        serial_conn (serial.Serial): The serial connection object for communicating with the glove.
        pub (rospy.Publisher): The ROS publisher for publishing glove data.
        print_data (DataPrint): The data structure holding sensor names and positions.
    Methods:
        __init__():
            Initializes the GloveNode, sets up parameters, serial connection, and ROS publisher.
        send_start_packet(packet_type):
            Sends a start streaming packet to the glove.
            Args:
                packet_type (int): The type of packet to send.
            Returns:
                int: 0 if the packet was successfully sent, 1 otherwise.
        send_packet(packet):
            Sends a packet to the glove.
            Args:
                packet (bytearray): The packet to send.
            Returns:
                int: 0 if the packet was successfully sent, 1 otherwise.
        read_data():
            Reads data from the glove, decodes it, and publishes it to the ROS topic.
        decode_sensor_values(data):
            Args:
                data (bytes): The raw data received from the glove.
            Returns:
                list[float]: A list of decoded sensor values.
        run():
            Main loop of the node. Sends start packets and continuously reads and publishes data.
    """
    __np_int16_dtype = np.dtype(np.int16).newbyteorder(
        ">"
    )  
    __np_int32_dtype = np.dtype(np.int32).newbyteorder(
        ">"
    ) 

    def __init__(self):
        rospy.init_node('data_getter', anonymous=True)

        self.__id = 0
        self.__packet_tick = 0
        
        # Read parameters
        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        self.baudrate = rospy.get_param('~baudrate', 230400)
        self.pub_rate = rospy.get_param('~rate', 1)
        
        # Serial connection
        self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
        
        # ROS Publisher
        self.pub = rospy.Publisher('glove_data', DataPrint, queue_size=1)
        # self.pub = rospy.Publisher('glove_joint_states', JointState, queue_size=10)
        
        # Sensor data
        # self.joint_state = JointState()
        # self.joint_state.name = [
        self.print_data = DataPrint()
        self.print_data.name = [
            # 'thumb_1', 'thumb_2', 'index_1', 'index_2',
            # 'middle_1', 'middle_2', 'ring_1', 'ring_2',
            # 'little_1', 'little_2', 'palm_arch', 'thumb_crossover',
            # 'abd_thumb', 'abd_index', 'abd_ring', 'abd_little',
            # 'press_thumb', 'press_index', 'press_middle',
            # 'press_ring', 'press_little'
            'thumb_2', 'thumb_1', 'index_2', 'index_1',
            'middle_2', 'middle_1', 'ring_2', 'ring_1',
            'little_2', 'little_1', 'palm_arch',
            'nop', 'thumb_crossover','nop', #nop to be deleted
            'press_thumb', 'press_index', 'press_middle',
            'press_ring', 'press_little',
            'abd_thumb', 'abd_index', 'abd_ring', 'abd_little'    
        ]
        # self.joint_state.position = [0.0] * len(self.joint_state.name)
        self.print_data.value = [0.0] * len(self.print_data.name) #np.zeros(23, dtype=np.int16)
        
        
        # self.lock = threading.Lock()
        # self.__sensors = [0.0] * 23  # replace 23 with actual sensor count
        
    def send_start_packet(self, packet_type):
        """
        Sends a start packet to initiate a specific operation.
        This method constructs a packet with a predefined structure and sends it
        using the `send_packet` method. The packet includes a header, command,
        payload length, packet type, checksum (BCC), and an end character.
        Args:
            packet_type (int): The type of packet to be sent, indicating the specific
                operation to start.
        Returns:
            bool: True if the packet was sent successfully, False otherwise.
        """
        
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
        Sends a packet of data through the serial connection.
        Args:
            packet (bytes): The data packet to be sent.
        Returns:
            int: Returns 0 if the packet was successfully sent, or 1 if there was 
                 an error (e.g., no serial connection or an exception occurred).
        Raises:
            Exception: If an error occurs during the sending process, it is caught 
                       and logged, but not re-raised.
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
        """
        Reads data from the serial connection, decodes it, and publishes the processed data.

        This method attempts to read data from the serial connection based on the expected
        length of joint state names. If data is successfully read, it decodes the sensor
        values and updates the `print_data` message with the decoded positions and the
        current timestamp. The updated message is then published to the associated ROS topic.

        In case of a serial communication error, an error message is logged.

        Raises:
            serial.SerialException: If there is an issue with the serial connection.

        Note:
            - The method assumes that `self.serial_conn` is a valid and open serial connection.
            - The `self.decode_sensor_values` method is used to process the raw data.
            - The `self.pub` publisher is used to publish the `print_data` message.
        """
        try:
            data = self.serial_conn.read(len(self.print_data.name)*2)  
            # rospy.loginfo(f"Received {len(data)} bytes: {data.hex()}")

            # rospy.loginfo(f"Raw data: {data}")
            # if self.serial_conn:
            #     rospy.loginfo("Serial connection is open")
            if data:
                #### JOINT STATE
                # self.joint_state.position = self.decode_sensor_values(data)
                # self.joint_state.header.stamp = rospy.Time.now()
                # self.pub.publish(self.joint_state)
                # rospy.loginfo(self.joint_state)
                #### PRINT DATA
                # self.print_data.value = self.decode_sensor_values(data)
                self.print_data.value = self.read_stream() #data
                
                self.print_data.header.stamp = rospy.Time.now()
                rospy.loginfo(self.print_data)
                self.pub.publish(self.print_data)
                # rospy.loginfo(self.print_data)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")
        

    def decode_sensor_values(self, data):
        """
        Decodes raw sensor data into a list of floating-point values.
        Args:
            data (iterable): The raw sensor data, expected to be an iterable of values 
                             that can be converted to floats.
        Returns:
            list: A list of floating-point values decoded from the input data.
        Note:
            This function assumes that the input data is already in a format that can 
            be directly converted to floats. Actual decoding logic may need to be 
            implemented based on the specific format of the raw sensor data.
        """
        
        # with self.lock:
            # Example: return dummy sensor values from data, you’ll need to implement actual decoding
        # rospy.loginfo(f"Raw data: {data}")
        return [int(i) for i in data]


    def read_stream(self):
        """
        Read incoming packets
        Set new_packet_available flag if a new packet has been received
        :return 0 if ok, 1 otherwise
        """
        # |PKT_HEADER|PKT_CMD_START|data_length|payload        |BCC|PKT_ENDCAR|
        #                                      |pkt_type|values|
        # |    1     |     1       |     1     |<data_length>-2| 1 |    1     |

        try:
            bcc = 0  # Block check character

            header_buffer = self.serial_conn.read(1)

            if not header_buffer or header_buffer[0] != PKT_HEADER:
                return 0

            bcc += header_buffer[0]

            # header found
            header_buffer = self.serial_conn.read(2)

            if not header_buffer or (header_buffer[0] != PKT_CMD_START_SAMPLING):
                return 0

            data_length = header_buffer[1]

            bcc += sum(header_buffer) & 0xFF  # mod 255
            bcc &= 0xFF

            # read data
            data_buffer = self.serial_conn.read(data_length)

            if len(data_buffer) != data_length:
                return 0

            # get payload
            payload_buffer = data_buffer[: data_length - 2]
            bcc += sum(payload_buffer) & 0xFF
            bcc &= 0xFF

            if bcc != data_buffer[data_length - 2]:  # check bcc
                return 0  # bcc doesn't match

            return np.copy(self._update_values(payload_buffer))
            # return np.copy(self.print_data.value) 

            # set flag, main application can read new values
            # self.__new_packet_available = True
        except Exception as e:
            print(e)
        #     return 1
        # else:
        #     self.pkt_count += 1

        # return 0

    def _update_values(self, packet):
        with self.lock:
            offset = 0
            if (pkt_type := packet[offset]) == PKT_TYPE_QUAT_FINGER:  # pkt_type
                offset += 1
                # ID
                size = 2
                count = 1
                self.__id = int.from_bytes(
                    packet[offset : offset + size], byteorder="big"
                )
                offset += count * size

                # Package Tick
                size = 4
                count = 1
                self.__packet_tick = int.from_bytes(
                    packet[offset : offset + size], byteorder="big"
                )
                offset += count * size

                # read 2 quaternions (8 32-bit values)
                # wrist and hand
                count = 8
                size = self.__np_int16_dtype.itemsize
                wrist_and_hands_quats = (
                    np.frombuffer(
                        packet, count=count, offset=offset, dtype=self.__np_int32_dtype
                    )
                    / 65536.0
                )
                self.__quat_wrist[:] = wrist_and_hands_quats[: count // 2]
                self.__quat_hand[:] = wrist_and_hands_quats[count // 2 :]
                offset += count * size

                # read 23 sensor values (23 2-Bytes values)
                count = 23
                size = self.__np_int16_dtype.itemsize
                self.__sensors[:] = np.frombuffer(
                    packet, count=count, offset=offset, dtype=self.__np_int16_dtype
                )

            elif pkt_type == PKT_TYPE_RAW_FINGER:
                pass

            # print("-----------------\n")
            # print(f"|{" ".join([f'{i:02x}' for i in packet])}|")
            # self.__print_values()
            # print("-----------------\n")

        return 0
    
    def run(self):
        """
        Executes the main loop for the data getter node.

        This method initializes the communication by sending start packets,
        then enters a loop where it continuously reads data and maintains
        a consistent publishing rate. The loop runs until the ROS node is
        shut down.

        Steps:
        1. Sends an initial start packet with a value of 0x0.
        2. Sleeps for a short duration to allow processing.
        3. Sends a second start packet with a value of 0x1.
        4. Enters a loop where it:
           - Reads data using the `read_data` method.
           - Sleeps to maintain the specified publishing rate.

        Note:
            The publishing rate is determined by the `self.pub_rate` attribute.

        Raises:
            rospy.ROSInterruptException: If the ROS node is interrupted during execution.
        """
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















##################################################################################################
##################################################################################################
##################################################################################################
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
