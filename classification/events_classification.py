# $ python -m spiking_arch.smnist_fp --resultroot spiking_arch/results/ --sron
# $ python -m spiking_arch.smnist_fp --dataroot "MNIST/" --resultroot "spiking_arch/results/spiking_act" --sron --batch 16

import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from classification.lsm import LiquidStateMachine
from classification.s_ron import SpikingRON
from classification.mixed_ron import MixedRON

from classification.utils import *
from classification.data_loader import *

import yaml

# Load YAML config
with open("/home/frankie/catkin_ws/src/dataglove/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Convert to a namespace to mimic argparse
args = argparse.Namespace(**config)


@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in data_loader:
        images = images.float().to(device)
        images = torch.flatten(images, start_dim=2)
        ys.append(labels.cpu()) 
        output = model(images)[0]
        activations.append(output[-1].cpu())
    activations = torch.cat(activations, dim=0).cpu().detach().numpy() # activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)



device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.cpu
    else torch.device("cpu")
)

n_inp = 1568

gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
epsilon = (
    args.epsilon - args.epsilon_range / 2.0,
    args.epsilon + args.epsilon_range / 2.0,
)

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):
    if args.sron:
        model = SpikingRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
        ).to(device)
    
    elif args.liquidron:
        model = LiquidRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            win_e=1,
            win_i=0.5,
            w_e=1,
            w_i=0.5,
            Ne=200,
            Ni=56,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device
        ).to(device)
    elif args.mixron:
        model = MixedRON(
            n_inp,
            args.n_hid,
            args.dt,
            gamma,
            epsilon,
            args.rho,
            args.inp_scaling,
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            args.perc,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=device,
        ).to(device) 
    else:
        raise ValueError("Wrong model choice.")
    

    train_loader, valid_loader, test_loader = get_data(
        args.dataroot, bs_train=16, bs_test=16, valid_perc=10.0
    )


    activations, ys = [], []
    
    print(train_loader.size)
    # for batch in next(iter(train_loader)):
    #     images, labels = batch[0].float(), batch[1].float() # Access only the first two items
    # for images, labels in train_loader:
    #     images = images.float().to(device)
    #     images = torch.flatten(images, start_dim=2)
        
    #     ys.append(labels.cpu()) 
        
    #     if args.liquidron:
    #         output, spk = model(images)
    #     else:
    #         output, velocity, u, spk = model(images) 
    #     activations.append(output[-1].cpu())
        

    
    # if args.liquidron:
    #     u= torch.stack(output)
    #     spk = torch.stack(spk)    
    #     plot_dynamics(u, spk, images, args.resultroot)
    # else:
    #     output = torch.stack(output)    
    #     spk = torch.stack(spk)    
    #     u = torch.stack(u)
    #     velocity = torch.stack(velocity)
    #     # print('shapes of tensors: \noutput: ', output.shape, '\nspikes: ', spk.shape, '\nmembrane potential: ', u.shape, '\nvelocity: ', velocity.shape, '\nx: ', images.shape)
    #     plot_dynamics(u, spk, images, args.resultroot, output=output, velocity=velocity)
    
    activations = torch.cat(activations, dim=0).detach().numpy() # activations = torch.cat(activations, dim=0).numpy()  
    ys = torch.cat(ys, dim=0).squeeze().numpy()
    # print("Activations shape:", activations.shape)
    # print("Labels shape:", ys.shape)  

    scaler = preprocessing.StandardScaler()
    
    activations = scaler.fit_transform(activations) 
    # activations = scaler.transform(activations) 
    classifier = LogisticRegression(max_iter=5000).fit(activations, ys)
    train_acc = test(train_loader, classifier, scaler)
    # valid_acc = test(valid_loader, classifier, scaler) #if not args.use_test else 0.0
    test_acc = test(test_loader, classifier, scaler) #if args.use_test else 0.0
    train_accs.append(train_acc)
    # valid_accs.append(valid_acc)
    test_accs.append(test_acc)
    print('Train accuracy: ', train_acc, '\nTest accuracy: ', test_acc)
simple_plot(train_accs, valid_accs, test_accs, args.resultroot)


if args.sron:
    f = open(os.path.join(args.resultroot, f"log_SRON{args.resultsuffix}.txt"), "a")
elif args.liquidron:
    f = open(os.path.join(args.resultroot, f"log_LiquidRON{args.resultsuffix}.txt"), "a")
elif args.mixron:
    f = open(os.path.join(args.resultroot, f"log_MixedRON{args.resultsuffix}.txt"), "a")
else:
    raise ValueError("Wrong model choice.")





ar = ""
for k, v in vars(args).items():
    ar += f"{str(k)}: {str(v)}, "
ar += (
    f"train: {[str(round(train_acc, 2)) for train_acc in train_accs]} "
    # f"valid: {[str(round(valid_acc, 2)) for valid_acc in valid_accs]} "
    f"test: {[str(round(test_acc, 2)) for test_acc in test_accs]}"
    f"mean/std train: {np.mean(train_accs), np.std(train_accs)} "
    # f"mean/std valid: {np.mean(valid_accs), np.std(valid_accs)} "
    f"mean/std test: {np.mean(test_accs), np.std(test_accs)}"
)
f.write(ar + "\n")
f.close()
