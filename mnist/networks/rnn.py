import torch

class RNNNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        torch.nn.Module.__init__(self)
        self.batch_size = kwargs["batch_size"]
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.i2r = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, recurrent):
        combined = torch.cat((input.reshape(-1, self.input_size), recurrent), 1)
        recurrent = self.i2r(combined)
        hidden = torch.sigmoid(self.i2h(combined))
        output = torch.sigmoid(self.h2o(hidden))
        return output, recurrent

    def initHidden(self, device):
        return torch.zeros(self.batch_size, self.hidden_size, device=device)
    
    def forward_batch(self, batch_data):
        hidden = self.initHidden(batch_data.device)
        batch_out = None
        for i in range(0, batch_data.shape[1]):
            batch_out, hidden = self(
                torch.tensor(batch_data[:,i], 
                dtype=torch.float32, device=batch_data.device
                ), hidden
            )
        return batch_out

    def evaluate_batch(self, batch_data, batch_target):
        hidden = self.initHidden(batch_data.device)
        out = None
        for i in range(0, batch_data.shape[1]):
            out, hidden = self(
                torch.tensor(batch_data[:,i], 
                dtype=torch.float32, device=batch_data.device
                ), hidden
            )

        return (out.max(1)[1] == (batch_target.to(batch_data.device))).sum().item()
