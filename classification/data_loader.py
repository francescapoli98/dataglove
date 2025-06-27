import os
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import traceback
import argparse
import numpy as np

import rosbag_pandas

import yaml
# Load YAML config
with open("/home/frankie/catkin_ws/src/dataglove/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)
# Convert to a namespace to mimic argparse
args = argparse.Namespace(**config)



class GloveDataset(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(columns=["time", "stiffness", "object"])
        self.labels = dataframe.apply(lambda row: ((row["stiffness"]*2) + row["object"]), axis=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values.astype(np.int16))
        y = torch.tensor(self.labels.iloc[idx].astype(np.int16))
        return x, y

######

def add_cols(folder_path, filename, column_name, keyword_mapping):
    file_path = os.path.join(folder_path, filename)
    for keyword, value in keyword_mapping.items():
        if keyword in filename:
            df = pd.read_csv(file_path)
            df[column_name] = value
            df.to_csv(file_path, index=False)
            # print(f"Updated {filename} with {column_name} = {value} because of keyword '{keyword}'")
            break  # Only one match expected


def cut_from_threshold(df, threshold_col, threshold_value):
    """
    Given a time series dataframe for a single object sample,
    cut from the first time the threshold_col exceeds threshold_value.
    More specifically, I want to consider the grasping as starting from the moment  
    the palm_arch exceeds 100 (is more than 0 and its near values, and spikes)
    """
    # Find the index where the threshold is first exceeded
    start_idx = df[df[threshold_col] >= threshold_value].index.min()
    
    # If no threshold is met, return empty or full df
    if pd.isna(start_idx):
        return pd.DataFrame()  # or: return df

    return df.loc[start_idx:].reset_index(drop=True)

def get_data(folder_path, bs_train=32, bs_test=32, valid_perc=10.0):
    csv_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.csv'):
            add_cols(folder_path, f, 'stiffness', args.stiffness)
            add_cols(folder_path, f, 'object', args.obj)
            csv_files.append(f)

    print(f"Found {len(csv_files)} CSV files")
    df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    # print(df_list)
    # for i, df in enumerate(df_list):
    #     print(f"--- File {i}: shape={df.shape} ---")
    #     print(df.head())

    combined_df = pd.concat(df_list, ignore_index=True)#.dropna()
    combined_df.drop(list(combined_df.filter(regex='header')), axis=1, inplace=True)
    combined_df.columns = combined_df.columns.str.removeprefix('.') 
    combined_df = cut_from_threshold(combined_df, 'palm_arch', 100)


    # print(combined_df.info())
    # print(combined_df.head())

    dataset = GloveDataset(combined_df)

    valid_size = int(len(dataset) * (valid_perc / 100.0))
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - valid_size - test_size

    print(f"Total samples: {len(dataset)} | Train: {train_size}, Valid: {valid_size}, Test: {test_size}")


    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)
    
    return train_loader, valid_loader, test_loader