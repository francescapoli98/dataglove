#!/usr/bin/env python3

import rospy
import torch
import os
import inspect
import numpy as np
from std_msgs.msg import Float32MultiArray
from sklearn.preprocessing import StandardScaler
from classification.mixed_ron import MixedRON

class MixedNode:
    def __init__(self):
        rospy.init_node("node_mixron", anonymous=True)

        self.batch_size = rospy.get_param('/dataglove_params/buffer_size', 50)
        self.input_size = 21
        self.start_param = rospy.get_param('/dataglove_params/start_mixron')
        self.started = False
        self.sub = None
        # Load model

        script_dir = os.path.dirname(os.path.abspath(__file__))  # path di node_mixedron.py
        ckpt_path = os.path.join(script_dir, "models", "mixedron_checkpoint.pt")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # checkpoint = torch.load("models/mixedron_checkpoint.pt", map_location='cpu')
        filtered_config = self.filter_model_args(MixedRON, checkpoint['config'])
        self.model = MixedRON(**filtered_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load or mock scaler
        self.scaler = StandardScaler()
        # if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
        #     self.scaler.mean_ = checkpoint['scaler_mean']
        #     self.scaler.scale_ = checkpoint['scaler_scale']
        # else:
        #     self.scaler.mean_ = np.zeros(self.input_size)
        #     self.scaler.scale_ = np.ones(self.input_size)

        # Timer to check for start signal
        rospy.Timer(rospy.Duration(0.5), self.check_start)

    def check_start(self, _):
        if not self.started and rospy.has_param(self.start_param):
            start_val = rospy.get_param(self.start_param)
            if start_val is True:
                rospy.loginfo(f"[MRON] Starting node after start_param {self.start_param}")
                self.started = True
                self.sub = rospy.Subscriber("/glove_buffer", Float32MultiArray, self.buffer_callback)
                rospy.delete_param(self.start_param)
    
    @staticmethod
    def filter_model_args(model_class, config_dict):
        # Get model init args except 'self'
        valid_args = inspect.signature(model_class.__init__).parameters.keys()
        valid_args = set(valid_args) - {'self'}

        return {k: v for k, v in config_dict.items() if k in valid_args}

    def buffer_callback(self, msg):
        try:
            flat_data = np.array(msg.data, dtype=np.float32)
            buffer = flat_data.reshape((self.batch_size, self.input_size))
            norm_data = self.scaler.transform(buffer)
            input_tensor = torch.tensor(norm_data, dtype=torch.float32)
            with torch.no_grad():
                prediction = self.model(input_tensor)
            rospy.loginfo(f"[MRON] Prediction: {prediction}")
        except Exception as e:
            rospy.logerr(f"[MRON] Error in callback: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MixedNode()
        node.run()
    except rospy.ROSInterruptException:
        pass