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
    tau = 5         # time constant in ms
    threshold = 0.5
    delta_t = torch.tensor(1)
    beta = torch.exp(-delta_t/torch.tensor(tau))


    num_classes = 26 # als neurons out
    num_neurons_in = 784
    num_neuro_hid = 1024

    mem_hidden = torch.zeros((steps, 1, num_neuro_hid))
    spk_hidden = torch.zeros((steps, 1, num_neuro_hid))
    mem_out = torch.zeros((steps, 1, num_classes))
    spk_out = torch.zeros((steps, 1, num_classes))

    mem_out_one = torch.zeros(steps + 1)
    spikes_out_one = torch.zeros(steps + 1)



    transf = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
    
    ds = get_emnist(transf, subset, batch_size)
    
    # try stuff with only one minibatch
    data = iter(ds)
    x, target = next(data)

    x = spikegen.latency(x[0], num_steps = steps, tau = tau, threshold = 0.01, clip = True, normalize = True, linear = True)
    
    
    target = target[0]
    x.reshape([steps, 1, 784])

    breakpoint()

    lif = snn.Leaky(threshold = threshold)

    con1 = torch.nn.Linear(284, 1024)
    lif1 = snn.Leaky(beta = beta)
    con2 = torch.nn.Linear(1024, 26)
    lif2 = snn.Leaky(beta = beta)

    mem1 = lif1.reset_mem()
    mem2 = lif2.reset_mem()


    # cur_in = torch.cat((torch.ones(1) * 5, torch.zeros(5), torch.ones(1), torch.zeros(5), torch.ones(1), torch.zeros(87)),0)

    for step in range(steps):
        cur = con1(x[step])
        spk_hidden[step+1], mem_hidden[step+1] = lif1(cur, mem_hidden[step]) # does this really work?
        cur = con2(spk_hidden[step])
        spk_out[step], mem_out[step] = lif2(cur, mem_out[step])



    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(x[:, 0].view(steps, -1), ax, s=25, c="black")

    plt.title("Input Layer")
    plt.suptitle(str(target[0]))
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
