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
        #grid search params
        leaky: float,
        # gamma: Union[float, Tuple[float, float]],
        # epsilon: Union[float, Tuple[float, float]],
        # gamma: float,  # Default value if not provided
        # gamma_range: float,  # Range for random sampling
        # epsilon: float,  # Default value if not provided
        # epsilon_range: float,  # Range for random sampling
        # rho: float,
        # input_scaling: float,
        # threshold: float,
        # resistance: float,
        # capacitance: float,
        # rc: float,
        # reset: float,
        # bias: float,
        # #no grid search
        # topology: Literal[
        #     "full", "lower", "orthogonal", "band", "ring", "toeplitz"
        # ] = "full",
        # reservoir_scaler=0.0,
        # sparsity=0.0,
        device="cpu",):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.leaky = leaky

        # Initialize layers
        self.fc1 = nn.Linear(self.n_inp, self.n_hid)
        self.lif1 = snn.Leaky(beta=self.leaky)
        self.fc2 = nn.Linear(self.n_hid, 1)
        self.lif2 = snn.Leaky(beta=self.leaky)

    def forward(self, x):
        # x = x.unsqueeze(1)  # shape: (batch, 1, input_dim)

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(1)):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return mem2_rec, spk2_rec #torch.stack(mem2_rec, dim=0),torch.stack(spk2_rec, dim=0),



# train_accs, valid_accs, test_accs = [], [], []
# for i in range(args.trials):

#     train_loader, valid_loader, test_loader = get_data(
#         args.dataroot, bs_train=16, bs_test=16, valid_perc=10.0
#     )

#     n_inp = train_loader.dataset[0][0].shape[0]  # Assuming the first dimension is the input size

#     # if args.sron:
#     model = SNNNet(
#         n_inp,
#         args.n_hid,
#         args.dt,
#         device=args.device
#         # device=device,
#     ).to(device)
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'config': args.__dict__,  # Converts Namespace to dict
#         }, "models/snn_checkpoint.pt")
        

#     activations, ys = [], []
    

#     for values, labels in train_loader:
#         values = values.float().to(args.device)
#         ys.append(labels.cpu()) 

#         output, spk = model(values)
#         activations.append(output[-1].cpu())
    
#     activations = torch.cat(activations, dim=0).detach().numpy() # activations = torch.cat(activations, dim=0).numpy()  
#     ys = torch.cat(ys, dim=0).squeeze().numpy()
#     # print("Activations shape:", activations.shape)
#     # print("Labels shape:", ys.shape)  

#     scaler = preprocessing.StandardScaler()
    
#     activations = scaler.fit_transform(activations) 
#     print("Activations after scaling:", activations.shape)
#     classifier = LogisticRegression(max_iter=5000).fit(activations, ys)
#     train_acc = test(train_loader, classifier, scaler)
#     # valid_acc = test(valid_loader, classifier, scaler) #if not args.use_test else 0.0
#     test_acc = test(test_loader, classifier, scaler) #if args.use_test else 0.0
#     train_accs.append(train_acc)
#     # valid_accs.append(valid_acc)
#     test_accs.append(test_acc)
#     print('Train accuracy: ', train_acc, '\nTest accuracy: ', test_acc)
# # simple_plot(train_accs, valid_accs, test_accs, args.resultroot)



# label_map = {
#     0: "soft-bottle",
#     1: "soft-ball",
#     2: "hard-bottle",
#     3: "hard-ball"
# }

# # Salva il dizionario label → nome
# with open("models/label_map.json", "w") as f:
#     json.dump(label_map, f)


# f = open(os.path.join(args.resultroot, f"SNN{args.resultsuffix}.txt"), "a")
# # Salva il classificatore
# dump(classifier, "models/snn_classifier.joblib")
# dump(scaler, "models/snn_scaler.joblib")


# ar = ""
# for k, v in vars(args).items():
#     ar += f"{str(k)}: {str(v)}, "
# ar += (
#     f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
#     # f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
#     f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
#     f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
#     # f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
#     f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
# )
# f.write(ar + "\n")
# f.close()

