# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import yaml
import rospy
from joblib import dump
import json
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression

# from classification.utils import *
# from classification.data_loader import *
# from classification.events_classification import test

# # Load YAML config
# with open("/home/frankie/catkin_ws/src/dataglove/config/params.yaml", "r") as f:
#     config = yaml.safe_load(f)
# # Convert to a namespace to mimic argparse
# model_config = config['model']
# rospy.loginfo(model_config)
# args = argparse.Namespace(**model_config)

# device = (
#     torch.device("cuda")
#     if torch.cuda.is_available() and not args.cpu
#     else torch.device("cpu")
# )

class SNNNet(nn.Module):
    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        leaky: float,
        device="cpu",):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.leaky = leaky
        # self.input_scaling = input_scaling

        # Initialize layers
        self.fc1 = nn.Linear(self.n_inp, self.n_hid)
        self.lif1 = snn.Leaky(beta=nn.Parameter(torch.tensor(self.leaky)))
        self.fc2 = nn.Linear(self.n_hid, 4)
        self.lif2 = snn.Leaky(beta=nn.Parameter(torch.tensor(self.leaky)))

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (batch, 1, input_dim)

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x[:, step])  # shape: (batch, n_hid)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return mem2_rec, spk2_rec 


