'''
Set of useful functions (mostly plots)
'''
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from typing import Literal



def simple_plot(train_accs: np.ndarray, valid_accs: np.ndarray, test_accs: np.ndarray, resultroot: str):
    print('this is the simple plot')
    # Convert lists to numpy arrays for easy manipulation
    train_accs = np.array(train_accs)
    valid_accs = np.array(valid_accs)
    test_accs = np.array(test_accs)
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot the mean accuracy for each stage (train, valid, test)
    plt.plot(np.arange(len(train_accs)), train_accs, label='Train Accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(np.arange(len(valid_accs)), valid_accs, label='Validation Accuracy', marker='s', linestyle='--', color='orange')
    plt.plot(np.arange(len(test_accs)), test_accs, label='Test Accuracy', marker='^', linestyle=':', color='green')
    
    # Add labels, title, and legend
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Trial for Each Experiment Stage')
    plt.legend()
    
    # Show grid for better readability
    plt.grid(True)
    
    # Save the figure as a PNG file
    plot_filename = os.path.join(resultroot, f"accuracy_plot.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    # Show the plot in a window
    # plt.show()
    
#######################################################################################################
#######################################################################################################
#######################################################################################################
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics(
    # activations: torch.Tensor,
    membrane_potential: torch.Tensor,
    spikes: torch.Tensor,
    x: torch.Tensor,
    resultroot: str,
    **kwargs  # Optional parameters (e.g., velocity)
):
    print('Plotting dynamics.')

    # Ensure input tensors are converted to numpy
    # activations = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
    membrane_potential = membrane_potential.detach().cpu().numpy() if isinstance(membrane_potential, torch.Tensor) else membrane_potential
    spikes = spikes.detach().cpu().numpy() if isinstance(spikes, torch.Tensor) else spikes
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    
    activations = kwargs.get("output", None)  # Optional parameter
    velocity = kwargs.get("velocity", None)  # Optional parameter
    
    if activations is not None:
        activations = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
    if velocity is not None:
        velocity = velocity.detach().cpu().numpy() if isinstance(velocity, torch.Tensor) else velocity
        
    # Handle optional parameters dynamically
    optional_params = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        optional_params[key] = value

    time_steps = np.arange(x.shape[1])
    # time_steps = np.arange(activations.shape[0])

    # Determine the number of subplots dynamically
    num_plots = 3 + len(optional_params)  # 3 mandatory plots (x, activations, spikes) + optional ones
    plt.figure(figsize=(8, 2 * num_plots))
    i=1
    # Plot input (x)
    plt.subplot(num_plots, 1, i)
    plt.title('Input (x)') #FOR N-MNIST: np.mean(x[0, :, :]) #also check for other datasets
    plt.plot(time_steps, np.mean(x[0, :, :], axis=1),
             #x[0, :, 0], 
             color="purple", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    i+=1

    # Plot activations
    if activations is not None:
        plt.subplot(num_plots, 1, i)
        plt.title('Hidden States (hy)')
        plt.plot(time_steps, activations[:, 0, 0], color="blue", linestyle='-', linewidth=1)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        i+=1
    

    # Plot velocity if provided
    if velocity is not None:
        plt.subplot(num_plots, 1, i)
        plt.title('Hidden States Derivatives (hz)')
        plt.plot(time_steps, velocity[:, 0, 0], color="orange", linestyle='-', linewidth=1)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        i+=1
        

    # Plot membrane potential
    # if membrane_potential is not None:
    plt.subplot(num_plots, 1, i)
    plt.title('Membrane Potential (u)')
    plt.plot(time_steps, membrane_potential[:, 0, 0], color="green", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    i+=1
        

    # Plot spikes
    plt.subplot(num_plots, 1, i)
    plt.title('Spiking times')
    spike_times = time_steps[spikes[:, 0, 0] == 1]
    spike_values = spikes[spikes[:, 0, 0] == 1, 0, 0]
    plt.scatter(spike_times, spike_values, color="red", zorder=5, s=30)
    plt.xlim(0, len(time_steps))
    plt.ylabel('Spike')
    

    plt.tight_layout()
    plt.savefig(f"{resultroot}/dynamics_plot.png")
    plt.close()
    plt.show()



def plot_dynam_mg(
    activations: torch.Tensor = None,
    # velocity: torch.Tensor = None,
    # times: int,
    membrane_potential: torch.Tensor = None,
    spikes: torch.Tensor = None,
    # x: torch.Tensor = None,
    resultroot: str = None,
):
    print('Plotting dynamic.')

    # Ensure the input tensors are in numpy format
    activations = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
    # velocity = velocity.detach().cpu().numpy() if isinstance(velocity, torch.Tensor) else velocity
    membrane_potential = membrane_potential.detach().cpu().numpy() if isinstance(membrane_potential, torch.Tensor) else membrane_potential
    spikes = spikes.detach().cpu().numpy() if isinstance(spikes, torch.Tensor) else spikes
    # print('activations: ', activations.size(), ' velocity: ', velocity.size(), ' u: ', membrane_potential.size(), ' spikes: ', spikes.size())
    # Get the time steps (assuming they are aligned with the tensor shapes)
    time_steps = np.arange(len(activations))#.shape[1])  # Number of time steps (length of time axis)
    print('Time steps shape: ', time_steps.shape)

    # Create a plot
    plt.figure(figsize=(8, 10))
    
    # Plot the images (x) 
    plt.subplot(2, 1, 1)
   
    plt.title('Membrane Potential (u)')
    # Plot the first hidden unit of the first channel
    plt.plot(time_steps, membrane_potential[:, 0, 0], label="Membrane Potential (u)", color="green", linestyle='-', linewidth=1)
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Plot the spikes (as vertical lines at spike times) - Selecting the first channel (layer) and first unit
    plt.subplot(2, 1, 2)
    plt.title('Spiking times')
    # Filter time steps where spikes == 1
    spike_times = time_steps[spikes[:, 0, 0] == 1]  # Time steps where spikes occur
    spike_values = spikes[spikes[:, 0, 0] == 1, 0, 0]  # Spike values (should be all 1s)

    # Scatter plot only the spikes == 1
    plt.scatter(spike_times,spike_values, color="red", label="Spikes", zorder=5, s=30)
    plt.xlim(0, len(time_steps))    
    plt.ylabel('Spike')
    plt.legend()


    # Finalize the plot
    plt.tight_layout()
    plt.savefig(f"{resultroot}/dynamics_plot.png")
    plt.close()
    plt.show()



def mg_results(target, predictions, std, resultroot, filename):    
    """
    Plots the target data vs. the predicted data.
    
    :param target: The ground truth target values (numpy array).
    :param predictions: The predicted values (numpy array).
    :param resultroot: Path to save the plot.
    :param filename: Name of the output file.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(target, label="Target Data", color='blue', linewidth=0.7)
    plt.plot(predictions, label="Predicted Data", color='red', linewidth=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Target vs Predicted Data")
    text = f"Mean Squared Error: {'%.3f'%std}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(resultroot, filename))
    plt.show()


# Example usage:
# plot_results(train_mse, test_mse, args.resultroot)


def acc_table(param_names, param_combinations, accuracies, resultroot, dataset):
    """
    Create a table to visualize parameter combinations and their corresponding accuracies.

    Args:
        param_combinations (list of tuples): List of parameter combinations.
        accuracies (list of float): List of accuracies corresponding to the parameter combinations.
    """
    # Create a DataFrame from parameter combinations and accuracies
    # param_names = ["dt", "threshold", "resistance", "capacitance", "reset"] #"rho", "inp_scaling",
    if dataset == 'mg':
        data = [list(comb) + [acc] for comb, acc in zip(param_combinations, accuracies)]
        df = pd.DataFrame(data, columns=param_names + ["MSE"])
        df = df.sort_values(by="MSE", ascending=True)
    else:    
        data = [list(comb) + [acc] for comb, acc in zip(param_combinations, accuracies)]
        df = pd.DataFrame(data, columns=param_names + ["Accuracy"])
        df = df.sort_values(by="Accuracy", ascending=False)

    # Create a table plot
    fig, ax = plt.subplots(figsize=(12, min(1 + len(df) * 0.3, 20)))  # Adjust height for the number of rows
    ax.axis("off")
    ax.axis("tight")

    # Use the DataFrame as the table content
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#add8e6"] * (len(param_names) + 1),  # Light background for column headers
    )
    table.auto_set_font_size(True)
    # table.set_fontsize(10)
    table.auto_set_column_width(range(len(df.columns)))
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight="bold")  # Make header text bold
        cell.set_edgecolor("gray")  # Add a border color
        # cell.set_height(0.05)  # Adjust height
        # cell.set_width(0.15)  # Adjust width


    # Adjust layout
    fig.tight_layout()
    # plt.savefig(f"{resultroot}/gridsearch_.png")
    plt.savefig("{}/gridsearch_{}.png".format(resultroot, dataset))
    plt.show()

def plot_hy(activations: torch.Tensor, x: torch.Tensor, resultroot: str):
    print("Plotting dynamics of the hidden state in comparison with the input.")

    # Convert tensors to numpy arrays if needed
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Validate dimensions
    assert activations.ndim == 3, f"Expected activations to be 3D, got {activations.ndim}D"
    assert x.ndim == 3, f"Expected x to be 3D, got {x.ndim}D"

    # Get time steps
    time_steps = np.arange(activations.shape[1])  # Time dimension is the second axis

    # Create plots
    plt.figure(figsize=(8, 10))

    # Plot input (x)
    plt.subplot(2, 1, 1)
    plt.title("Input (x)")
    plt.plot(time_steps, x[0, :, 0], label="Input Data", color="purple", linestyle="-", linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    # Plot hidden states (hy)
    plt.subplot(2, 1, 2)
    plt.title("Hidden States (hy)")
    plt.plot(time_steps, activations[0, :, 0], label="Hidden State", color="blue", linestyle="-", linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    # Finalize
    plt.tight_layout()
    os.makedirs(resultroot, exist_ok=True)
    plt.savefig(f"{resultroot}/hy_plot.png")
    plt.close()
    print(f"Plot saved to {resultroot}/hy_plot.png")

    
def plot_accuracy_fluctuations(param_combinations, accuracies):
    """
    Plots accuracy fluctuations for each set of values in the grid search.

    Args:
        param_combinations (list of tuples): The parameter combinations tested in the grid search.
        accuracies (list of floats): The validation accuracies corresponding to each parameter combination.
    """
    # Ensure the inputs are of the same length
    assert len(param_combinations) == len(accuracies), "Parameter combinations and accuracies must have the same length"

    # Create a unique identifier for each parameter combination
    param_labels = [" | ".join(f"{k}={v}" for k, v in zip(param_names, params)) for params in param_combinations]

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o', linestyle='-', label='Validation Accuracy')

    # Label the points with their parameter combinations (optional for small grids)
    for i, label in enumerate(param_labels):
        plt.annotate(label, (i, accuracies[i]), fontsize=8, rotation=45, alpha=0.7, ha='right')

    plt.title("Accuracy Fluctuations Across Grid Search")
    plt.xlabel("Parameter Set Index")
    plt.ylabel("Validation Accuracy")
    plt.xticks(range(len(accuracies)), labels=range(len(accuracies)), rotation=45)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


##########################################
# FROM ACDS.ARCHETYPES.UTILS

def count_parameters(model):
    """Return total number of parameters and
    trainable parameters of a PyTorch model.
    """
    params = []
    trainable_params = []
    for p in model.parameters():
        params.append(p.numel())
        if p.requires_grad:
            trainable_params.append(p.numel())
    pytorch_total_params = sum(params)
    pytorch_total_trainableparams = sum(trainable_params)
    return pytorch_total_params, pytorch_total_trainableparams


def sparse_eye_init(M: int) -> torch.FloatTensor:
    """Generates an M x M matrix to be used as sparse identity matrix for the re-scaling
    of the sparse recurrent kernel in presence of non-zero leakage. The neurons are
    connected according to a ring topology, where each neuron receives input only from
    one neuron and propagates its activation only to one other neuron. All the non-zero
    elements are set to 1.

    Args:
        M (int): number of hidden units.

    Returns:
        torch.FloatTensor: MxM identity matrix.
    """
    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """Generates an M x N matrix to be used as sparse (input) kernel For each row only C
    elements are non-zero (i.e., each input dimension is projected only to C neurons).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1]

    Args:
        M (int): number of hidden units
        N (int): number of input units
        C (int): number of nonzero elements

    Returns:
        torch.FloatTensor: MxN dense matrix
    """
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype("f") - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1) -> torch.FloatTensor:
    """Generates an M x M matrix to be used as sparse recurrent kernel. For each column
    only C elements are non-zero (i.e., each recurrent neuron take sinput from C other
    recurrent neurons). The non-zero elements are generated randomly from a uniform
    distribution in [-1,1].

    Args:
        M (int): number of hidden units
        C (int): number of nonzero elements

    Returns:
        torch.FloatTensor: MxM dense matrix
    """
    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype("f") - 1)
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def spectral_norm_scaling(
    W: torch.FloatTensor, rho_desired: float
) -> torch.FloatTensor:
    """Rescales W to have rho(W) = rho_desired .

    Args:
        W (torch.FloatTensor): input matrix to be rescaled
        rho_desired (float): desired spectral radius

    Returns:
        torch.FloatTensor: rescaled matrix
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)


def get_hidden_topology(
    n_hid: int,
    topology: Literal["full", "lower", "orthogonal", "band", "ring", "toeplitz"],
    sparsity: float,
    scaler: float,
) -> torch.FloatTensor:
    """Generates the hidden-to-hidden weight matrix according to the specified topology
    and sparsity.

    Args:
        n_hid (int): number of hidden units.
        topology (str): topology of the hidden-to-hidden weight matrix. Options
            are 'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz'.
        sparsity (float): sparsity of the hidden-to-hidden weight matrix.
        scaler (float): scaling factor for the hidden-to-hidden weight matrix.

    Returns:
        torch.Tensor: hidden-to-hidden weight matrix.
    """

    def get_sparsity(A):
        n_hid = A.shape[0]
        sparsity = 100 * (n_hid**2 - np.count_nonzero(A)) / n_hid**2
        return sparsity
    assert sparsity >= 0 and sparsity < 1, "Sparsity must be in [0,1)"

    if topology == "full":
        h2h = 2 * (2 * torch.rand(n_hid, n_hid) - 1)
    elif topology == "lower":
        h2h = torch.tril(2 * torch.rand(n_hid, n_hid) - 1)
        if sparsity > 0:
            n_zeroed_diagonals = int(sparsity * n_hid)
            for i in range(n_hid - 1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == "orthogonal":
        rand = torch.rand(n_hid, n_hid)
        orth = torch.linalg.qr(rand)[0]
        identity = torch.eye(n_hid)
        if sparsity > 0:
            n_zeroed_rows = int(sparsity * n_hid)
            idxs = torch.randperm(n_hid)[:n_zeroed_rows].tolist()
            identity[idxs, idxs] = 0.0
        h2h = torch.matmul(identity, orth)
        get_sparsity(h2h.numpy())
    elif topology == "band":
        h2h = 2 * torch.rand(n_hid, n_hid) - 1
        if sparsity > 0:
            n_zeroed_diagonals = int(np.sqrt(sparsity) * n_hid)
            for i in range(n_hid - 1, n_hid - n_zeroed_diagonals - 1, -1):
                h2h.diagonal(-i).zero_()
                h2h.diagonal(i).zero_()
        get_sparsity(h2h.numpy())
    elif topology == "ring":
        # scaler = 1
        h2h = torch.zeros(n_hid, n_hid)
        for i in range(1, n_hid):
            h2h[i, i - 1] = 1
        h2h[0, n_hid - 1] = 1
        h2h = scaler * h2h
        get_sparsity(h2h.numpy())
    elif topology == "toeplitz":
        from scipy.linalg import toeplitz

        bandwidth = int(scaler)  # 5
        upperdiagcoefs = np.zeros(n_hid)
        upperdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs = np.zeros(n_hid)
        lowerdiagcoefs[:bandwidth] = 2 * torch.rand(bandwidth) - 1
        lowerdiagcoefs[0] = upperdiagcoefs[0]  # diagonal coefficient
        h2h = toeplitz(list(lowerdiagcoefs), list(upperdiagcoefs))
        get_sparsity(h2h)
        h2h = torch.Tensor(h2h)
    else:
        raise ValueError(
            "Invalid topology. Options are 'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz'"
        )
    return h2h