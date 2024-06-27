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

import snn_model_01


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,7), sharex=True, 
                            gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title("Metrics")

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(101, -1), ax[1], s = 0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(101, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 26])

    plt.show()

def plot_metrics(epochs, train_loss, test_loss, train_acc, test_acc):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
    
    fig.set_size_inches(10, 5)
        
    ax[0].plot(range(len(epochs)), train_loss)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy in %')
    ax[0].set_title('Accuracy of the different models')
    ax[0].legend(title = '[Layers] Embedding')
    
    
    ax[1].plot(range(len(epochs)), test_loss)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss of the different models')
    ax[1].legend(title = '[Layers] Embedding')

    plt.show()



def get_emnist_letters(
        transform: Any = transforms.ToTensor(), 
        target_transform: Any = transforms.ToTensor(),
        subset: int = None, 
        batch_size: int = 128
    ) -> object:
    '''Function to get the letters of the EMNIST dataset - like MNIST, just with letters'''
    
    train = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = True,
        download = True,
        transform = transform,
        target_transform = target_transform
    )

    test = datasets.EMNIST(
        root = 'data',
        split = 'letters',
        train = False,
        download = True,
        transform = transform,
        target_transform = target_transform
    )
    
    if subset:
        train = snntorch.utils.data_subset(train, subset)
        test  = snntorch.utils.data_subset(test, subset)


    train = torch.utils.data.DataLoader(
        train, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = None, 
        num_workers = 0 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    test = torch.utils.data.DataLoader(
        test, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        prefetch_factor = None, 
        num_workers = 0 # num_workers should be threads (-1, to leave some thread for other programs)
    ) 

    return train, test



if __name__ == "__main__":
    
    subset = 10
    batch_size = 1024
    epochs = 2

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


    #basic preprocessing
    transf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.Lambda(lambda x: x.reshape(num_neuro_in))
    ])

    #rescales targets from indices 1-26 to indices 0-25 (otherwise error)
    target_transf = transforms.Compose([
        transforms.Lambda(lambda x: x -1 )
    ])
    
    #get data
    train, test = get_emnist_letters(
        transform = transf, 
        target_transform = target_transf,
        subset = subset, 
        batch_size = batch_size
    )
    
    # try stuff with only one minibatch
    # data = iter(train)
    # x, target = next(data)
    # target = target[0]
    # x = x.reshape([steps, batch_size, num_neuro_in]).to(torch.int8)
    # lif = snn.Leaky(threshold = threshold)
    # plot_snn_spikes(x,spk_hidden, spk_out, 'something')


    breakpoint()
    model = snn_model_01.SNN(
        layers = [num_neuro_in, num_neuro_hid, num_classes],
        beta = beta,
        # spike_grad = snntorch.surrogate.FastSigmoid(),
        num_steps = steps,
        threshold = threshold,
        tau = tau
    )
    model.set_optimiser()
    model.set_loss(snntorch.functional.loss.ce_temporal_loss())
    model.training_loop(train, test, epochs)

    #model.plot_metrics(epochs = epochs, train_loss = model.train_loss, test_loss = model.test_loss, train_acc = 0, test_acc = 0)

    # would it work to write this into the class?
    # train_acc = model.train_acc()
    # print(train_acc)

    
    breakpoint()
