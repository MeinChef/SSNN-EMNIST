import numpy as np
import matplotlib.pyplot as plt
from typing import Any

import snntorch
import snntorch.functional
import snntorch.spikegen
import snntorch.spikeplot as splt
import snntorch.surrogate
import snntorch.utils

import torch
import torch.utils
import torch.utils.data

from torchvision import datasets, transforms

import snn_model


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,7), sharex=True, 
                            gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(101, -1), ax[1], s = 0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(101, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 26])

    plt.show()


def get_emnist_letters(
        transform: Any = transforms.ToTensor(), 
        subset: int = None, 
        batch_size: int = 128,
        threshold: float = 0.01,
        tau: int = 5,
        num_steps: int = 100
    ) -> object:
    '''Function to get the letters of the EMNIST dataset - like MNIST, just with letters'''
    
    train = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = True,
        download = True,
        transform = transform
    )

    test = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = False,
        download = True,
        transform = transform
    )
    
    if subset:
        train = snntorch.utils.data_subset(train, subset)
        test  = snntorch.utils.data_subset(test, subset)


    train = torch.utils.data.DataLoader(
        train, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = 4, 
        num_workers = torch.get_num_threads() - 1 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    test = torch.utils.data.DataLoader(
        test, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = 4, 
        num_workers = torch.get_num_threads() - 1 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    return train, test



if __name__ == "__main__":
    
    subset = 10
    batch_size = 1024

    steps = 100     # simulation time steps
    tau = 5         # time constant in ms
    threshold = 0.5
    delta_t = torch.tensor(1)
    beta = torch.exp(-delta_t / torch.tensor(tau)) # no idea why this is the correct beta current, but the documentation said so


    num_neuro_in = 784 # input features, 784 = 28*28
    num_neuro_hid = 1024
    num_classes = 26 # also neurons out

    # mem_hidden = torch.zeros((steps + 1, batch_size, num_neuro_hid))
    # spk_hidden = torch.zeros((steps + 1, batch_size, num_neuro_hid))
    # mem_out = torch.zeros((steps + 1, batch_size, num_classes))
    # spk_out = torch.zeros((steps + 1, batch_size, num_classes))



    transf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.Lambda(lambda x: x.reshape(num_neuro_in))
    ])
    
    train, test = get_emnist_letters(
        transform = transf, 
        subset = subset, 
        batch_size = batch_size,
        threshold = threshold, 
        tau = tau,
        num_steps = steps
    )
    
    # try stuff with only one minibatch
    # data = iter(train)
    # x, target = next(data)
    # target = target[0]
    # x = x.reshape([steps, batch_size, num_neuro_in]).to(torch.int8)
    # lif = snn.Leaky(threshold = threshold)
    # plot_snn_spikes(x,spk_hidden, spk_out, 'something')
    mini = 1000
    maxi = 0
    for _, target in train:
        mini = min(mini, target.min().item())
        maxi = max(maxi, target.max().item())    

    # min should be 0
    # max 25
    breakpoint()
    model = snn_model.SNN(
        layers = [num_neuro_in, num_neuro_hid, num_classes],
        beta = beta,
        # spike_grad = snntorch.surrogate.FastSigmoid(),
        num_steps = steps,
        threshold = threshold,
        tau = tau
    )
    model.set_optimiser()
    model.set_loss(snntorch.functional.loss.ce_temporal_loss())
    model.training_loop(train, test, 1)

    
    breakpoint()
