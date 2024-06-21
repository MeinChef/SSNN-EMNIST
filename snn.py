import numpy as np
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import utils, spikegen
import snntorch.spikeplot as splt

import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms



def create_model() -> object:
    pass

def get_emnist(transf = transforms.ToTensor(), subset: int = 10, batch_size: int = 128) -> object:
    '''Function to get the letters of the EMNIST dataset - like MNIST, just with letters'''
    
    ds = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = True,
        download = True,
        transform = transf
    )

    ds = utils.data_subset(ds, subset)

    ds = torch.utils.data.DataLoader(ds, batch_size = batch_size, shuffle = True)

    return ds



if __name__ == "__main__":
    
    subset = 10
    batch_size = 128

    steps = 100     # simulation time steps
    tau = 5         # time constant
    num_classes = 26


    transf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
    
    ds = get_emnist(transf, subset, batch_size)
    
    # try stuff with only one minibatch
    data = iter(ds)
    x, target = next(data)

    x = spikegen.latency(x, num_steps = steps, tau = tau, threshold = 0.01, clip = True, normalize = True, linear = True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(x[:, 0].view(steps, -1), ax, s=25, c="black")

    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
