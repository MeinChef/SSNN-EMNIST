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
            num_steps: int = 100,
            threshold: float = 0.01,
            tau: int = 5
        ) -> None:

        super().__init__()

        self.num_steps = num_steps
        self.threshold = threshold
        self.tau = tau
        self.num_layer = len(layers) - 1

        assert len(layers) == 3, 'currently this supports only 1 hidden and one output layer'

        self.fc1 = torch.nn.Linear(layers[0], layers[1])
        self.lif1 = snntorch.Leaky(beta = beta)
        self.fc2 = torch.nn.Linear(layers[1], layers[2])
        self.lif2 = snntorch.Leaky(beta = beta)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: make return spikes in a way that the optimizer is able to distinguish which class should be there with what probability.
        #       IDEA: the neuron that spikes first determines the class - one could take the max classes and substract the idx of the spiking neuron, then softmax
        
        
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
            
    def training_loop(
            self, 
            train: Any = None, 
            test: Any = None, 
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

                if target.item() < 26:
                    loss_val = self.loss(mem_rec, target)

                if target.item() == 0: print('yei')
                if target.item() == 26: continue
                
                # loss_val = self.loss(mem_rec, target)


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
                    continue

        
    
    def set_optimiser(self, optim: Any = torch.optim.Adam, learning_rate: float = 0.001) -> None:
        self.optimiser = optim(self.parameters(), lr = learning_rate)

    def set_loss(self, loss: Any = torch.nn.CrossEntropyLoss()) -> None:
        self.loss = loss

