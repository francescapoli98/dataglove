#!/usr/bin/env python3

import rospy
import torch
import os
import traceback
import inspect
import numpy as np
from joblib import load
import json
from dataglove.msg import ClassificationData  # replace with your actual msg name
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from classification.snn import SNNNet
import rospkg

class SNNNode:
    def __init__(self):
        self.batch_size = rospy.get_param('/dataglove_params/buffer_size', 50)
        self.input_size = 21
        self.started = False
        self.mix_subscriber = None

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('dataglove')
        model_dir = os.path.join(pkg_path, 'models')

        ckpt_path = os.path.join(model_dir, "snn_checkpoint.pt")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        filtered_config = self.filter_model_args(SNNNet, checkpoint['config'])
        self.model = SNNNet(**filtered_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.scaler = load(os.path.join(model_dir, "snn_scaler.joblib"))
        self.classifier = load(os.path.join(model_dir, "snn_classifier.joblib"))

        with open(os.path.join(model_dir, "label_map.json"), "r") as f:
            self.label_map = json.load(f)

        self.log_file_path = os.path.join(os.path.expanduser('~'), 'snn_latency_log.txt')

        # Subscribe to your custom message with header and data[]
        self.mix_subscriber = rospy.Subscriber("/glove_buffer", ClassificationData, self.buffer_callback)

    @staticmethod
    def filter_model_args(model_class, config_dict):
        valid_args = set(inspect.signature(model_class.__init__).parameters.keys()) - {'self'}
        return {k: v for k, v in config_dict.items() if k in valid_args}

    def buffer_callback(self, msg):
        try:
            rospy.loginfo(f"[SNN] Starting node")

            # Calculate latency
            now = rospy.Time.now()
            latency = (now - msg.header.stamp).to_sec() * 1000.0  # in milliseconds
            rospy.loginfo(f"[SNN] Message latency: {latency:.2f} ms")
            with open(self.log_file_path, 'a') as f:
                f.write(f"{latency:.3f}\n")

            flat_data = np.array(msg.data, dtype=np.float32)
            buffer = flat_data.reshape((self.batch_size, self.input_size))

            palm_arch_values = buffer[:, 10]
            if not np.any(palm_arch_values > 0):
                rospy.loginfo(f"[SNN] Waiting for grasping to start")
                return

            input_tensor = torch.tensor(buffer, dtype=torch.float32)

            with torch.no_grad():
                model_output = self.model(input_tensor)[0]
                if isinstance(model_output, list):
                    model_output = model_output[0]

                output = model_output[-1, :]  # (256,)
                output_np = output.numpy().reshape(1, -1)

                norm_output = self.scaler.transform(output_np)

                pred = self.classifier.predict(norm_output)[0]
                label = self.label_map.get(str(pred), pred)
                rospy.loginfo(f"[SNN] Predicted label: {label}")

        except Exception as e:
            rospy.logerr(f"[SNN] Error in callback: {e}")
            rospy.logerr("[SNN] Full traceback:\n%s", traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node("node_snn")
        node = SNNNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

# import rospy
# import torch
# import os
# import traceback
# import inspect
# import numpy as np
# from joblib import load
# import json
# from std_msgs.msg import Float32MultiArray, Int16MultiArray
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from classification.snn import SNNNet
# import rospkg

# class SNNNode:
#     def __init__(self):
#         self.batch_size = rospy.get_param('/dataglove_params/buffer_size', 50)
#         self.input_size = 21
#         self.started = False
#         self.mix_subscriber = None

#         # Trova la cartella del pacchetto e la directory dei modelli
#         rospack = rospkg.RosPack()
#         pkg_path = rospack.get_path('dataglove')
#         model_dir = os.path.join(pkg_path, 'models')

#         # Carica il modello MixedRON
#         ckpt_path = os.path.join(model_dir, "snn_checkpoint.pt")
#         checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#         filtered_config = self.filter_model_args(SNNNet, checkpoint['config'])
#         self.model = SNNNet(**filtered_config)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()

#         # Carica scaler e classificatore
#         self.scaler = load(os.path.join(model_dir, "snn_scaler.joblib"))
#         self.classifier = load(os.path.join(model_dir, "snn_classifier.joblib"))

#         # Carica il dizionario delle label
#         with open(os.path.join(model_dir, "label_map.json"), "r") as f:
#             self.label_map = json.load(f)

#         # Subscriber ROS
#         self.mix_subscriber = rospy.Subscriber("/glove_buffer", Int16MultiArray, self.buffer_callback)

#     @staticmethod
#     def filter_model_args(model_class, config_dict):
#         valid_args = set(inspect.signature(model_class.__init__).parameters.keys()) - {'self'}
#         return {k: v for k, v in config_dict.items() if k in valid_args}

#     def buffer_callback(self, msg):
#         try:
#             rospy.loginfo(f"[SNN] Starting node")

#             flat_data = np.array(msg.data, dtype=np.float32)
#             # rospy.loginfo(f"Palm_arch:{msg.data.palm_arch}")
#             buffer = flat_data.reshape((self.batch_size, self.input_size))
#             # Wait to classify if palm_arch (index X) <= 10
#             palm_arch_values = buffer[:, 10]

#             if not np.any(palm_arch_values > 0):
#                 rospy.loginfo(f"[SNN] Waiting for grasping to start")
#                 return
#             input_tensor = torch.tensor(buffer, dtype=torch.float32)

#             with torch.no_grad():
#                 model_output = self.model(input_tensor)[0]
#                 if isinstance(model_output, list):
#                     model_output = model_output[0]

#                 output = model_output[-1, :]  # (256,)
#                 output_np = output.numpy().reshape(1, -1)  # shape: (1, 256)

#                 # Ora normalizza le feature (non i dati grezzi!)
#                 norm_output = self.scaler.transform(output_np)

#                 # Classifica
#                 pred = self.classifier.predict(norm_output)[0]
#                 label = self.label_map[str(pred)] if str(pred) in self.label_map else pred
#                 rospy.loginfo(f"[SNN] Predicted label: {label}")

#         except Exception as e:
#             rospy.logerr(f"[SNN] Error in callback: {e}")
#             rospy.logerr("[SNN] Full traceback:\n%s", traceback.format_exc())


#     def run(self):
#         rospy.spin()

# if __name__ == '__main__':
#     try:
#         rospy.init_node("node_snn")
#         node = SNNNode()
#         node.run()
#     except rospy.ROSInterruptException:
#         pass
