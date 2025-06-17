#!/usr/bin/env python3

import rospy
import torch
import os
import inspect
import numpy as np
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from sklearn.preprocessing import StandardScaler
from classification.s_ron import SpikingRON

class SRONNode:
    def __init__(self):
        rospy.init_node("node_sron", anonymous=True)

        self.batch_size = rospy.get_param('/dataglove_params/buffer_size', 50)
        self.input_size = 21
        # self.start_param = rospy.get_param('/dataglove_params/start_sron')
        self.started = False
        self.sron_subscriber = None
        # Load model

        script_dir = os.path.dirname(os.path.abspath(__file__))  # path di node_sron.py
        ckpt_path = os.path.join(script_dir, "models", "sron_checkpoint.pt")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # checkpoint = torch.load("models/sron_checkpoint.pt", map_location='cpu')
        filtered_config = self.filter_model_args(SpikingRON, checkpoint['config'])
        self.model = SpikingRON(**filtered_config)#**checkpoint['config']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load or mock scaler
        self.scaler = StandardScaler()
        self.sron_subscriber = rospy.Subscriber("/glove_buffer", Int16MultiArray, self.buffer_callback)
        # if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
        #     self.scaler.mean_ = checkpoint['scaler_mean']
        #     self.scaler.scale_ = checkpoint['scaler_scale']
        # else:
        #     self.scaler.mean_ = np.zeros(self.input_size)
        #     self.scaler.scale_ = np.ones(self.input_size)

        # Timer to check for start signal
        # rospy.Timer(rospy.Duration(0.5), self.check_start)

    # def check_start(self, _):
    #     if not self.started and rospy.has_param(self.start_param):
    #         start_val = rospy.get_param(self.start_param)
    #         if start_val is True:
    #             # rospy.loginfo(f"[SRON] Starting node after start_param {self.start_param}")
    #             self.started = True
    #             self.sub = rospy.Subscriber("/glove_buffer", Int16MultiArray, self.buffer_callback)
    #             rospy.delete_param(self.start_param)

    @staticmethod
    def filter_model_args(model_class, config_dict):
        # Get model init args except 'self'
        valid_args = inspect.signature(model_class.__init__).parameters.keys()
        valid_args = set(valid_args) - {'self'}

        final_args = {k: v for k, v in config_dict.items() if k in valid_args}
        # rospy.loginfo(f"[SRON] Filtered model args: {final_args}")
        return final_args

    def buffer_callback(self, msg):
        try:
            rospy.loginfo(f"[SRON] Starting node")
            flat_data = np.array(msg.data, dtype=np.float32)
            buffer = flat_data.reshape((self.batch_size, self.input_size))
            norm_data = self.scaler.transform(buffer)
            input_tensor = torch.tensor(norm_data, dtype=torch.float32)
            with torch.no_grad():
                prediction = self.model(input_tensor)
            rospy.loginfo(f"[SRON] Prediction: {prediction}")
        except Exception as e:
            rospy.logerr(f"[SRON] Error in callback: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SRONNode()
        node.run()
    except rospy.ROSInterruptException:
        pass