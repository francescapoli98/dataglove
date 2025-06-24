# $ python -m spiking_arch.smnist_fp --resultroot spiking_arch/results/ --sron
# $ python -m spiking_arch.smnist_fp --dataroot "MNIST/" --resultroot "spiking_arch/results/spiking_act" --sron --batch 16

import argparse
import os
import warnings
import numpy as np
import torch
import rospy
from joblib import dump
import json
import torch.nn.utils
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from classification.lsm import LiquidStateMachine
from classification.s_ron import SpikingRON
from classification.mixed_ron import MixedRON
from classification.snn import SNNNet


from classification.utils import *
from classification.data_loader import *

import yaml

# Load YAML config
with open("/home/frankie/catkin_ws/src/dataglove/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)
# Convert to a namespace to mimic argparse
model_config = config['model']
# rospy.loginfo(model_config)
args = argparse.Namespace(**model_config)


@torch.no_grad()
def test(data_loader, classifier, scaler):
    activations, ys = [], []
    for images, labels in data_loader:
        images = images.float().to(args.device)
        # images = torch.flatten(images, start_dim=2)
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

# gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
# epsilon = (
#     args.epsilon - args.epsilon_range / 2.0,
#     args.epsilon + args.epsilon_range / 2.0,
# )

train_accs, valid_accs, test_accs = [], [], []
for i in range(args.trials):

    train_loader, valid_loader, test_loader = get_data(
        args.dataroot, bs_train=128, bs_test=128, valid_perc=10.0
    )

    n_inp = train_loader.dataset[0][0].shape[0]  # Assuming the first dimension is the input size

    if args.sron:
        model = SpikingRON(
            n_inp,
            args.n_hid,
            args.dt,
            args.gamma,
            args.gamma_range,
            args.epsilon,
            args.epsilon_range,
            # gamma,
            # epsilon,
            args.rho,
            args.input_scaling,
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=args.device
            # device=device,
        ).to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': args.__dict__,  # Converts Namespace to dict
            }, "models/sron_checkpoint.pt")

    
    elif args.liquidron:
        model = LiquidStateMachine(
            n_inp,
            args.n_hid,
            args.dt,
            # gamma,
            # epsilon,
            args.gamma,
            args.gamma_range,
            args.epsilon,
            args.epsilon_range,
            args.rho,
            args.input_scaling,
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
            device=args.device
            # device=device
        ).to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': args.__dict__,  # Converts Namespace to dict
            }, "models/lsm_checkpoint.pt")

    elif args.mixron:
        model = MixedRON(
            n_inp,
            args.n_hid,
            args.dt,
            args.gamma,
            args.gamma_range,
            args.epsilon,
            args.epsilon_range,
            args.rho,
            args.input_scaling,
            args.threshold,
            args.rc,
            args.reset,
            args.bias,
            args.perc,
            topology=args.topology,
            sparsity=args.sparsity,
            reservoir_scaler=args.reservoir_scaler,
            device=args.device
        ).to(device) 
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': args.__dict__,  # Converts Namespace to dict
            }, "models/mixedron_checkpoint.pt")
    elif args.snn:
            # if args.sron:
        model = SNNNet(
            n_inp,
            args.n_hid,
            args.leaky,
            # args.input_scaling,
            device=args.device
            # device=device,
        ).to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': args.__dict__,  # Converts Namespace to dict
            }, "models/snn_checkpoint.pt")
    else:
        raise ValueError("Wrong model choice.")
    


    activations, ys = [], []
    

    for values, labels in train_loader:
        values = values.float().to(args.device)
        # images = torch.flatten(images, start_dim=2)
        ys.append(labels.cpu()) 
        # print(values.size())

        
        if (args.liquidron or args.snn):
            output, spk = model(values)
        else:
            output, velocity, u, spk = model(values) 
        activations.append(output[-1].cpu())
        

    
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
    print("Activations shape:", activations.shape)
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
# simple_plot(train_accs, valid_accs, test_accs, args.resultroot)



label_map = {
    0: "soft-bottle",
    1: "soft-ball",
    2: "hard-bottle",
    3: "hard-ball"
}

# Salva il dizionario label → nome
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)


if args.sron:
    f = open(os.path.join(args.resultroot, f"log_SRON{args.resultsuffix}.txt"), "a")
    # Salva il classificatore
    dump(classifier, "models/sron_classifier.joblib")
    dump(scaler, "models/sron_scaler.joblib")

elif args.liquidron:
    f = open(os.path.join(args.resultroot, f"log_LiquidRON{args.resultsuffix}.txt"), "a")
    # Salva il classificatore
    dump(classifier, "models/lsm_classifier.joblib")
    dump(scaler, "models/lsm_scaler.joblib")

elif args.mixron:
    f = open(os.path.join(args.resultroot, f"log_MixedRON{args.resultsuffix}.txt"), "a")
    # Salva il classificatore
    dump(classifier, "models/mron_classifier.joblib")
    dump(scaler, "models/mron_scaler.joblib")
elif args.snn:
    f = open(os.path.join(args.resultroot, f"SNN{args.resultsuffix}.txt"), "a")
    # Salva il classificatore
    dump(classifier, "models/snn_classifier.joblib")
    dump(scaler, "models/snn_scaler.joblib")

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

