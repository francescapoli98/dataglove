import math
import threading
import numpy as np
import serial
import transforms3d.euler as euler


# Packet Fields
PKT_TYPE_NONE = 0x0  # stop streaming
PKT_TYPE_QUAT_FINGER = 0x1  # start streaming quaternion and calibrated sensor values
PKT_TYPE_RAW_FINGER = 0x3  # start streaming raw values

PKT_HEADER = ord("$")
PKT_ENDCAR = ord("#")

PKT_CMD_UPDATE_VIBRO = 0x60
PKT_CMD_START_SAMPLING = 0x0A

# RIGHT HAND
# sensor positions in sensors array
ThumbPh1R = 1  # thumb 1st phalange
ThumbPh2R = 0  # thumb 2nd phalange
IndexPh1R = 3  # index first phalange
IndexPh2R = 2  # index second phalange
MiddlePh1R = 5  # index first phalange
MiddlePh2R = 4  # index second phalange
RingPh1R = 7  # index first phalange
RingPh2R = 6  # index second phalange
LittlePh1R = 9  # index first phalange
LittlePh2R = 8  # index second phalange

# palm arch and cross over sensors
PalmArchR = 10
ThumbCrossOverR = 12

# abduction sensors
AbdThumbR = 19
AbdIndexR = 20
AbdRingR = 21
AbdLittleR = 22

# pressure sensors
PressThumbR = 14
PressIndexR = 15
PressMiddleR = 16
PressRingR = 17
PressLittleR = 18


class VMG30:

    __np_int32_dtype = np.dtype(np.int32).newbyteorder(
        ">"
    )  # big endian int 32 data type
    __np_int16_dtype = np.dtype(np.int16).newbyteorder(
        ">"
    )  # big endian int 16 data type

    def __init__(self, port="/dev/ttyUSB0"):
        self.__device: serial.Serial = None
        self.__connected = False
        self.port = port
        self.__new_packet_available = False
        self.__id = 0
        self.__packet_tick = 0

        self.__sensors = np.zeros(23, dtype=np.int16)

        self.__quat_wrist = np.zeros(4, dtype=np.float32)
        self.__quat_hand = np.zeros(4, dtype=np.float32)

        self.lock = threading.RLock()

        self.pkt_count = 0

    def open_device(self):
        print(f"Open serial port {self.port}")
        try:
            self.__device = serial.Serial(
                self.port,
                baudrate=230400,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                timeout=0.1,
            )
            if self.__device:
                self.__connected = True

        except (ValueError, serial.SerialException) as e:
            print(f"Error opening serial port: {e}")
            return 1

    def close_device(self):
        """Close device if opened"""

        print(f"Close serial port {self.port}")
        if self.__connected:
            self.__device.close()
        else:
            print(f"Serial port {self.port} already closed")

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
        if not self.__connected:
            return 1

        try:
            self.__device.write(packet)
            print(f"Sent {len(packet)} bytes")

            return 0
        except Exception as e:
            print(f"Error Sending a packet: {e}")
            return 1

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

            header_buffer = self.__device.read(1)

            if not header_buffer or header_buffer[0] != PKT_HEADER:
                return 0

            bcc += header_buffer[0]

            # header found
            header_buffer = self.__device.read(2)

            if not header_buffer or (header_buffer[0] != PKT_CMD_START_SAMPLING):
                return 0

            data_length = header_buffer[1]

            bcc += sum(header_buffer) & 0xFF  # mod 255
            bcc &= 0xFF

            # read data
            data_buffer = self.__device.read(data_length)

            if len(data_buffer) != data_length:
                return 0

            # get payload
            payload_buffer = data_buffer[: data_length - 2]
            bcc += sum(payload_buffer) & 0xFF
            bcc &= 0xFF

            if bcc != data_buffer[data_length - 2]:  # check bcc
                return 0  # bcc doesn't match

            self._update_values(payload_buffer)

            # set flag, main application can read new values
            self.__new_packet_available = True
        except Exception as e:
            print(e)
            return 1
        else:
            self.pkt_count += 1

        return 0

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
                size = self.__np_int32_dtype.itemsize
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

    def update_vibro(self, vibro):
        """
        Update vibro-tactile engines power

        :param vibro = new vibro-tactile engines values (sequence of 5 values)

        RIGHT GLOVE:
        vibro[0] = thumb;
        vibro[1] = index;
        vibro[2] = middle;
        vibro[3] = ring;
        vibro[4] = little;

        LEFT GLOVE:
        vibro[4] = thumb;
        vibro[3] = index;
        vibro[2] = middle;
        vibro[1] = ring;
        vibro[0] = little;
        """
        payload_length = 0x08  # send 8 bytes
        buffer = bytearray(
            [
                PKT_HEADER,
                PKT_CMD_UPDATE_VIBRO,
                payload_length,
                *vibro[:5],
                vibro[4],  # this is replicated but not used
                0x00,
                PKT_ENDCAR,
            ]
        )

        buffer[9] = sum(buffer[:9]) & 0xFF  # bcc of first 8 bytes

        return self.send_packet(buffer)

    @staticmethod
    def quat2rpy(quat):
        """
        Compute roll pitch and yaw angles from quaternion values (radians)
        """
        # TODO R and P are flipped vs the original C++ version!!!!!
        return np.degrees(euler.quat2euler(quat, "sxyz"))

    @staticmethod
    def compute_rpy(quat):
        # original from cpp sources

        rpy = np.zeros(3, dtype=np.float32)

        ratio = 180.0 / math.pi
        rpy[0] = -ratio * math.atan2(
            2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
        )
        rpy[1] = -ratio * math.asin(2.0 * (quat[0] * quat[2] - quat[3] * quat[1]))
        rpy[2] = ratio * math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return rpy

    @property
    def rpy_wrist(self):
        """
        Get a copy of the wrist orientation in euler angles in degrees
        """
        with self.lock:
            return self.quat2rpy(self.__quat_wrist)

    @property
    def rpy_hand(self):
        """
        Get a copy of the hand orientation in euler angles in degrees
        """
        with self.lock:
            return self.quat2rpy(self.__quat_hand)

    def reset_new_packet(self):
        """
        Reset new available packet flag
        """
        with self.lock:
            self.__new_packet_available = False

    def is_new_packet_available(self):
        with self.lock:
            return self.__new_packet_available

    @property
    def packet_tick(self):
        with self.lock:
            return self.__packet_tick

    def get_sensor_value(self, index):
        """
        Get sensor value
        :param index index of the requested finger
        :return sensor measure
        :raises ValueError
        """

        if 0 <= index < 23:
            raise ValueError("No such sensor")

        with self.lock:
            return self.__sensors[index]

    @property
    def sensors(self):
        """
        Get a copy of the sensors array
        """
        with self.lock:
            return np.copy(self.__sensors)

    @property
    def quaternion_wrist(self):
        """
        Get a copy of the wrist orientation in quaternion
        """
        with self.lock:
            return np.copy(self.__quat_wrist)

    @property
    def quaternion_hand(self):
        """
        Get a copy of the hand orientation in quaternion
        """
        with self.lock:
            return np.copy(self.__quat_hand)

    def __print_values(self):
        print("Package Tick:", self.__packet_tick)
        print("Thumb values:", self.__sensors[ThumbPh1R], self.__sensors[ThumbPh2R])
        print("Index values:", self.__sensors[IndexPh1R], self.__sensors[IndexPh2R])
        print("Middle values:", self.__sensors[MiddlePh1R], self.__sensors[MiddlePh2R])
        print("Ring values:", self.__sensors[RingPh1R], self.__sensors[RingPh2R])
        print("Little values:", self.__sensors[LittlePh1R], self.__sensors[LittlePh2R])

        rpy_wrist = self.quat2rpy(self.__quat_wrist)
        print(
            f"Wrist rpy: " f"{rpy_wrist[0]:.3f} {rpy_wrist[1]:.3f} {rpy_wrist[2]:.3f}"
        )
        rpy_hand = self.quat2rpy(self.__quat_hand)
        print(f"Hand rpy: " f"{rpy_hand[0]:.3f} {rpy_hand[1]:.3f} {rpy_hand[2]:.3f}")

        print(
            f"Wrist quat: "
            f"{self.__quat_wrist[0]:.3f} {self.__quat_wrist[1]:.3f} {self.__quat_wrist[2]:.3f} {self.__quat_wrist[3]:.3f}"
        )
        print(
            f"Hand quat: "
            f"{self.__quat_hand[0]:.3f} {self.__quat_hand[1]:.3f} {self.__quat_hand[2]:.3f} {self.__quat_hand[3]:.3f}"
        )
