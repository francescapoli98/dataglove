import os
import csv
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from vmg30 import *


def get_data(
    root: os.PathLike, bs_train: int, bs_test: int):

    # folder_path = './logs'
    # Optional: filter CSVs that contain "session" in the name ### to be used for a specific object but not necessary
    # csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'session' in f]

    
    dataframes = []
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Combine all into one DataFrame
    dataset = pd.concat(dataframes, ignore_index=True)


    # Split into train/validation/test (DVSGesture does not have predefined splits)
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