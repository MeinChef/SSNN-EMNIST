import torch
import torch.utils.data
import snntorch
import snntorch.spikegen
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

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

        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        train_accs = []
        train_losss = []
        test_accs = []
        test_losss = []


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        for epoch in range(epochs):
            for x, target in train:
                x.to(device)
                target.to(device)

                # latency encoding of inputs
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

                # calculate loss
                loss_val = self.loss(mem_rec, target)

                # metrics
                train_loss.append(loss_val.item())
                #####train_acc.append(np.mean(np.argmax(x, -1) == np.argmax(target, -1)))
                train_acc.append(self.calc_accuracy(spk_rec, target))

                # clear prev gradients, calculate gradients, weight update
                self.optimiser.zero_grad()
                loss_val.backward()
                self.optimiser.step()
            
                                  

        
            # test loop
            with torch.no_grad():
                self.eval()

                for x, target in test:
                    x.to(device)
                    target.to(device)

                    # latency encoding of inputs
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
                    spk_rec, mem_rec = self(x)

                    # calculate loss
                    loss_val = self.loss(mem_rec, target)

                    # metrics
                    test_loss.append(loss_val.item())
                    test_acc.append(self.calc_accuracy(spk_rec, target))

            
            train_accs.append(np.mean(train_acc))
            train_losss.append(np.mean(train_loss))
            test_accs.append(np.mean(test_acc))
            test_losss.append(np.mean(test_loss))



            print(f'Train Losses after Epoch {epoch}: {train_losss}')
            print(f'Test Losses after Epoch {epoch}: {test_losss}')
            print(f'Train Accs after Epoch {epoch}: {train_accs}')
            print(f'Test Accs after Epoch {epoch}: {test_accs}')
            print(epochs)

        self.plot_metrics(epochs = epochs,
                            train_loss = train_losss, 
                            test_loss = test_losss,
                            train_acc = train_accs,
                            test_acc = test_accs)
                    

        
    
    def set_optimiser(self, optim: Any = torch.optim.Adam, learning_rate: float = 0.001) -> None:
        self.optimiser = optim(self.parameters(), lr = learning_rate)

    def set_loss(self, loss: Any = torch.nn.CrossEntropyLoss()) -> None:
        self.loss = loss

    def calc_accuracy(self, spk_rec, targets):
        _, idx = spk_rec.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

        return acc

    def plot_metrics(self, epochs, train_loss, test_loss, train_acc, test_acc):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
        
        fig.set_size_inches(10, 5)
            
        ax[0].plot(range(epochs), train_acc, label = "Training")
        ax[0].plot(range(epochs), test_acc, label = "Testing")
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy in %')
        ax[0].set_title('Accuracy')
        ax[0].legend()
        
        ax[1].plot(range(epochs), train_loss, label = "Training")
        ax[1].plot(range(epochs), test_loss, label = "Testing")
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss')
        ax[1].legend()

        plt.show()