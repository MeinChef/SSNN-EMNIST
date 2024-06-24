import torch
import torch.utils.data
import snntorch
import snntorch.spikegen
from typing import Any

class SNN(torch.nn.Module):
    def __init__(
            self, 
            layers: list = [784,1024,26], 
            beta: torch.Tensor = torch.tensor(0.8), 
            spike_grad: Any = snntorch.surrogate.FastSigmoid(),
            num_steps: int = 100,
            threshold: float = 0.01,
            tau: int = 5
        ) -> None:

        super().__init__()

        self.num_steps = num_steps
        self.threshold = threshold
        self.tau = tau
        self.num_layer = len(layers) - 1

        for idx in range(len(layers) - 1):
            # this is black magic and bad practice and never should be used, but we need this because torch is fucking picky.
            exec(f'self.con{idx} = torch.nn.Linear(layers[idx], layers[idx + 1])')
            exec(f'self.lay{idx} = snntorch.Leaky(beta = beta, spike_grad = spike_grad)')

        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        rec_mem = torch.empty(size = (self.num_layer, self.num_steps + 1), dtype = torch.float32)
        rec_spk = torch.empty(size = (self.num_layer, self.num_steps + 1), dtype = torch.int8)
        spk = x

        for step in range(self.num_steps):
            
            for idx in range(self.num_layer):
                exec(f'rec_mem[{idx}, step] = self.lay{idx}.reset_mem()')
                exec(f'cur = self.con{idx}(spk)')
                exec(f'rec_spk[{idx}, step + 1], rec_mem[{idx}, step + 1] = self.lay{idx}(cur, rec_mem[{idx}, step])')

        # TODO: make return spikes in a way that the optimizer is able to distinguish which class should be there with what probability.
        #       IDEA: the neuron that spikes first determines the class - one could take the max classes and substract the idx of the spiking neuron, then softmax
        return rec_spk[-1,1:], rec_mem[-1,1:]
    
    def train(
            self, 
            train: list[torch.Tensor, torch.utils.data.DataLoader] = None, 
            test: list[torch.Tensor, torch.utils.data.DataLoader] = None, 
            epochs: int = 25
        ) -> tuple[torch.Tensor, torch.Tensor]:

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        for epoch in range(epochs):
            for x, target in train:
                x.to(device)
                target.to(device)

                x = snntorch.spikegen.latency(
                    data = x, 
                    num_steps = self.num_steps, 
                    threshold = self.threshold,
                    tau = self.tau, 
                    clip = True, 
                    normalize = True, 
                    linear = True
                )
                
                # forward pass
                self.train()
                spk_rec, mem_rec = self(x)

                loss_val = torch.zeros((1), dtype = torch.float, device = device)
                for step in range(self.num_steps):
                    loss_val += self.loss(mem_rec[step], target)


                self.optimiser.zero_grad()
                loss_val.backward()
                self.optimiser.step()
            

        
            # test loop
            with torch.no_grad():
                for x, target in test:
                    x.to(device)
                    target.to(device)

                    x = snntorch.spikegen.latency(
                        data = x, 
                        num_steps = self.num_steps, 
                        threshold = self.threshold,
                        tau = self.tau, 
                        clip = True, 
                        normalize = True, 
                        linear = True
                    )
                    pass

        
    
    def set_optimiser(self, optim: Any = torch.optim.Adam, learning_rate: float = 0.001) -> None:
        self.optimiser = optim(self.parameters(), lr = learning_rate)

    def set_loss(self, loss: Any = torch.nn.CrossEntropyLoss()) -> None:
        self.loss = loss

