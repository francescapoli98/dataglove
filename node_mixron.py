#!/usr/bin/env python3

import rospy
import torch
import os
import traceback
import inspect
import numpy as np
from joblib import load
import json
import csv
from dataglove.msg import ClassificationData  # replace with your actual msg name
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from classification.mixed_ron import MixedRON
import rospkg

class MixedNode:
    def __init__(self):
        self.batch_size = rospy.get_param('/dataglove_params/buffer_size', 50)
        self.input_size = 21
        self.started = False
        self.mix_subscriber = None

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('dataglove')
        model_dir = os.path.join(pkg_path, 'models')

        ckpt_path = os.path.join(model_dir, "mixedron_checkpoint.pt")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        filtered_config = self.filter_model_args(MixedRON, checkpoint['config'])
        self.model = MixedRON(**filtered_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.scaler = load(os.path.join(model_dir, "mron_scaler.joblib"))
        self.classifier = load(os.path.join(model_dir, "mron_classifier.joblib"))

        with open(os.path.join(model_dir, "label_map.json"), "r") as f:
            self.label_map = json.load(f)

        self.log_file_path = os.path.join(os.path.expanduser('~'), 'mron_latency_log.txt')
        self.predtime_file_path = os.path.join(os.path.expanduser('~'), 'mron_predict_lat.txt')
        self.last_label = None  # Store last prediction

        # Subscribe to your custom message with header and data[]
        self.mix_subscriber = rospy.Subscriber("/glove_buffer", ClassificationData, self.buffer_callback)
        rospy.loginfo(f"[MRON] Starting node")


    @staticmethod
    def filter_model_args(model_class, config_dict):
        valid_args = set(inspect.signature(model_class.__init__).parameters.keys()) - {'self'}
        return {k: v for k, v in config_dict.items() if k in valid_args}
    
    @staticmethod
    def append_to_csv(pred, label, file_path):
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([pred, label])

    def buffer_callback(self, msg):
        try:

            # Calculate latency
            now = rospy.Time.now()
            latency = (now - msg.header.stamp).to_sec() * 1000.0  # in milliseconds
            # rospy.loginfo(f"[SNN] Message latency: {latency:.2f} ms")
            with open(self.log_file_path, 'a') as f:
                f.write(f"{latency:.3f}\n")

            flat_data = np.array(msg.data, dtype=np.float32)
            buffer = flat_data.reshape((self.batch_size, self.input_size))

            palm_arch_values = buffer[:, 10]
            if not np.any(palm_arch_values > 0):
                # rospy.loginfo(f"[MRON] Waiting for grasping to start")
                return

            input_tensor = torch.tensor(buffer, dtype=torch.float32)

            with torch.no_grad():
                model_output = self.model(input_tensor)[0]
                if isinstance(model_output, list):
                    model_output = model_output[0]

                output = model_output[-1, :]  # (256,)
                output_np = output.numpy().reshape(1, -1)

                norm_output = self.scaler.transform(output_np)
                ##start time
                before_pred = rospy.Time.now()
                pred = self.classifier.predict(norm_output)[0]
                ## end time
                after_pred = rospy.Time.now()
                pred_time = (after_pred - before_pred).to_sec() * 1000.0  # in milliseconds
                with open(self.predtime_file_path, 'a') as f:
                    f.write(f"{(pred_time):.3f}\n")
                
                label = self.label_map.get(str(pred), pred)
                self.append_to_csv(pred, label, os.path.join(os.path.expanduser('~'), 'MRON_predictions.csv'))
                if str(label) != str(self.last_label):
                    rospy.loginfo(f"[MRON] Predicted label: {label}")
                self.last_label = label  # Update last prediction

        except Exception as e:
            rospy.logerr(f"[MRON] Error in callback: {e}")
            rospy.logerr("[MRON] Full traceback:\n%s", traceback.format_exc())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node("node_mixron")
        node = MixedNode()
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
# from classification.mixed_ron import MixedRON
# import rospkg

# class MixedNode:
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
#         ckpt_path = os.path.join(model_dir, "mixedron_checkpoint.pt")
#         checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#         filtered_config = self.filter_model_args(MixedRON, checkpoint['config'])
#         self.model = MixedRON(**filtered_config)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()

#         # Carica scaler e classificatore
#         self.scaler = load(os.path.join(model_dir, "mron_scaler.joblib"))
#         self.classifier = load(os.path.join(model_dir, "mron_classifier.joblib"))

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
#             rospy.loginfo(f"[MRON] Starting node")

#             # Dati grezzi dal messaggio
#             flat_data = np.array(msg.data, dtype=np.float32)
#             buffer = flat_data.reshape((self.batch_size, self.input_size))
#             palm_arch_values = buffer[:, 10]

#             if not np.any(palm_arch_values > 0):
#                 rospy.loginfo(f"[MRON] Waiting for grasping to start")
#                 return

#             # Converti in tensore senza normalizzarlo prima
#             input_tensor = torch.tensor(buffer, dtype=torch.float32)

#             # Estrai le feature dal modello
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
#                 rospy.loginfo(f"[MRON] Predicted label: {label}")

#         except Exception as e:
#             rospy.logerr(f"[MRON] Error in callback: {e}")
#             rospy.logerr("[MRON] Full traceback:\n%s", traceback.format_exc())


#     def run(self):
#         rospy.spin()

# if __name__ == '__main__':
#     try:
#         rospy.init_node("node_mixron")

#         node = MixedNode()
#         node.run()
#     except rospy.ROSInterruptException:
#         pass
