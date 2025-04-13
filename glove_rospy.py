#!/usr/bin/python3
import rospy
import serial
import numpy as np
from sensor_msgs.msg import JointState

class GloveNode:
    def __init__(self):
        rospy.init_node('glove_node', anonymous=True)
        
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
        
    def read_data(self):
        """Reads data from the glove and updates joint states."""
        try:
            data = self.serial_conn.read(50)  # Read enough bytes for a packet
            if data:
                self.joint_state.position = self.decode_sensor_values(data)
                self.joint_state.header.stamp = rospy.Time.now()
                self.pub.publish(self.joint_state)
        except serial.SerialException as e:
            rospy.logerr(f"Serial error: {e}")
        
    def decode_sensor_values(self, data): 
        """Decodes raw data from the glove into sensor values."""
        # Placeholder: Convert data to meaningful values
        # return np.random.rand(len(self.joint_state.name)).tolist()  # Replace with real decoding
        
    
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
