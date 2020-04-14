import torch
import torch.nn as nn


class GRU(nn.Module):
    
    def __init__(self, D_in, H, D_out, n_layers, dropout = 0.2):
        super(GRU, self).__init__()
        self.hidden_size = H
        self.n_layers = n_layers

        self.gru = nn.GRU(D_in, H, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(H, D_out)
        self.relu = nn.ReLU()
    
    def forward(self, x, h):
        out, h = self.gru(x,h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden



