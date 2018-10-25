import torch

class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        torch.nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def forward_batch(self, batch_data):
        batch_out = self(batch_data.view(-1, batch_data.shape[1], batch_data.shape[2]).type(torch.float32))
        return batch_out

    def evaluate_batch(self, batch_data, batch_target):
        out = self(batch_data.view(-1, batch_data.shape[1], batch_data.shape[2]).type(torch.float32))
        return (out.max(1)[1] == (batch_target.to(batch_data.device))).sum().item()
