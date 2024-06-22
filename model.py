import torch
import snntorch as snn


class SNN(torch.nn.Module):
    def __init__(self, layers: list = [784,1024,26], beta: torch.Tensor = torch.tensor(0.8)):
        super.__init__()

        # self.layers = []
        # for idx in range(len(layers) - 1):
        #     self.layers.append(torch.nn.Linear(layers[idx], layers[idx+1]))
        #     self.layers.append(snn.Leaky(beta = beta))

        con1 = torch.nn.Linear(layers[0], layers[1])
        lif1 = snn.Leaky(beta = beta)
        con2 = torch.nn.Linear(layers[1], layers[2])
        lif2 = snn.Leaky(beta = beta)

        mem1 = lif1.reset_mem()
        mem2 = lif2.reset_mem()
        
