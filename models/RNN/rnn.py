import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class BitCounterRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.1):
        super(BitCounterRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        # Handle packed sequences
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            # Get the data from packed sequence and reshape
            x_data = x.data.unsqueeze(-1)  # Add feature dimension
            # Create new packed sequence with reshaped data
            x = torch.nn.utils.rnn.PackedSequence(
                x_data,
                x.batch_sizes,
                x.sorted_indices,
                x.unsorted_indices
            )
        else:
            # If not packed, reshape regular tensor
            x = x.unsqueeze(-1)  # Add feature dimension

        # Initialize hidden state
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            batch_size = x.batch_sizes[0]  # Get the batch size from the first batch
        else:
            batch_size = x.size(0)
            
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device if isinstance(x, torch.nn.utils.rnn.PackedSequence) else x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # If packed sequence, unpack it and get last outputs
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            out, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # Get the last output for each sequence
            idx = (lengths - 1).to(out.device)
            batch_size = out.size(0)
            out = out[torch.arange(batch_size).to(out.device), idx]
        else:
            out = out[:, -1, :]
        
        out = self.norm(out)
        out = self.fc(out)
        
        return out