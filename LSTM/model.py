import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        hidden = (
            torch.randn(1, x.size(0), self.hidden_size),
            torch.randn(1, x.size(0), self.hidden_size)
        )
        
        out, _ = self.lstm(x, hidden)
        
        out = self.fc(out[:, -1, :])
        return out.view(-1)