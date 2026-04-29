import torch
import torch.nn as nn

class LSTMRNN(nn.Module):
    """
    A standard LSTM-based RNN component used across TimeGAN modules.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = None, use_sigmoid: bool = False):
        super(LSTMRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_dim = output_dim
        self.use_sigmoid = use_sigmoid
        
        if output_dim is not None:
            self.linear = nn.Linear(hidden_dim, output_dim)
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
            
    def forward(self, x):
        # x is (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        
        if self.output_dim is not None:
            out = self.linear(out)
            
        if self.use_sigmoid:
            out = self.sigmoid(out)
            
        return out
