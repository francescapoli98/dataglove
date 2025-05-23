import os
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import rospy
import rosbag
import traceback 
import yaml
from classification.utils import *

class GloveDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        values = torch.tensor(row[1:].values, dtype=torch.float32)  # joint values
        label = torch.tensor(0)  # Replace with actual label if available
        return values, label




def get_data(root: os.PathLike, bs_train: int, bs_test: int, valid_perc: float = 10.0):
    # IF YOU WANT TO CONVERT .bag FILES TO .csv FILES, UNCOMMENT THE FOLLOWING LINES
    # Convert all .bag files to .csv if not already done
    # for file in os.listdir(root):
    #     if file.endswith(".bag"):
    #         bag_path = os.path.join(root, file)
    #         bag_to_csv(bag_path, root)

    # 2. Load all .csv files into pandas DataFrames
    dataframes = []
    for file in os.listdir(root):
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, header=None)
            dataframes.append(df)

    # 3. Combine into one dataset
    dataset_df = pd.concat(dataframes, ignore_index=True)

    # 4. Create custom PyTorch dataset
    dataset = GloveDataset(dataset_df)

    # 5. Split into train/val/test
    valid_size = int(len(dataset) * (valid_perc / 100.0))
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - valid_size - test_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)

    return train_loader, valid_loader, test_loader





###############################################################################################
###############################################################################################
# def bag_to_csv(bag_path, output_dir):
#     """
#     Converts a .bag file with messages containing:
#     - Header header
#     - string[] name
#     - int16[] value

#     into a CSV file with:
#     - Timestamps as row headers
#     - Joint names as columns (dynamically updated)
#     """
#     output_filename = os.path.splitext(os.path.basename(bag_path))[0] + ".csv"
#     csv_file = os.path.join(output_dir+'/dataset/', output_filename)

#     try:
#         with rosbag.Bag(bag_path, 'r') as bag:
#             data_rows = []
#             joint_names_set = set()

#             for _, msg, t in bag.read_messages():
#                 timestamp = t.to_sec()
#                 name_value_dict = dict(zip(msg.name, msg.value))
#                 joint_names_set.update(name_value_dict.keys())
#                 data_rows.append((timestamp, name_value_dict))
#                 print(f"Timestamp: {timestamp}, Names: {name_value_dict.keys()}")

#             joint_names = sorted(joint_names_set)

#             with open(csv_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['timestamp'] + joint_names)

#                 for timestamp, name_value_dict in data_rows:
#                     row = [timestamp]
#                     for joint in joint_names:
#                         row.append(name_value_dict.get(joint, ''))  # Fill missing with ''
#                     writer.writerow(row)

#         # print(f"[INFO] Wrote CSV: {csv_file}")

#     except Exception as e:
#         print(f"[ERROR] Failed to process bag file: {e}")
#         traceback.print_exc()
###############################################################################################
