import torch
import torchsnn

class StigmergicNetwork(torchsnn.StigmergicModule):
    def __init__(self, *modules, **kwargs):
        torchsnn.StigmergicModule.__init__(self)
        torchsnn.Sequential(target=self, *modules)
        self.lastlayer = kwargs["lastlayer"] if "lastlayer" in kwargs else -1

    def time_forward(self, time_input):
        self.reset()
        out = None
        for i in range(0, time_input.shape[1]):
            out = self.forward(
                torch.tensor(time_input[:,i],
                    dtype=torch.float32,
                    device=time_input.device
                ),
                lastlayer=self.lastlayer if i != time_input.shape[1] - 1 else -1
            )
            self.tick()
        return out
    
    def forward_batch(self, batch_data):
        return self.time_forward(batch_data)

    def evaluate_batch(self, batch_data, batch_target):
        out = self.time_forward(batch_data)
        return (out.max(1)[1] == (batch_target.to(batch_data.device))).sum().item()

