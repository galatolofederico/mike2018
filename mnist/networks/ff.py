import torch


class FFNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        torch.nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.add_module("input", torch.nn.Linear(input_size, hidden_size))
        self.add_module("activation_input", torch.nn.Sigmoid())
        for i in range(0, num_layers):
            self.add_module("l"+str(i), torch.nn.Linear(hidden_size, hidden_size))
            self.add_module("a"+str(i), torch.nn.Sigmoid())
        self.add_module("output", torch.nn.Linear(hidden_size, num_classes))
        self.add_module("activation_output", torch.nn.Sigmoid())
    
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def forward_batch(self, batch_data):
        batch_out = self(batch_data.view(-1, batch_data.shape[1] * batch_data.shape[2]).type(torch.float32))
        return batch_out
    
    def evaluate_batch(self, batch_data, batch_target):
        out = self(batch_data.view(-1, batch_data.shape[1] * batch_data.shape[2]).type(torch.float32))
        return (out.max(1)[1] == (batch_target.to(batch_data.device))).sum().item()
