"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class LSTMBaseline(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_hidden, num_outputs, num_layers):
        super(LSTMBaseline, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_hidden,
                            num_layers=num_layers)

        self.out = nn.Linear(num_hidden, num_outputs)

        # The hidden state is a learned parameter
        if torch.cuda.is_available():
            self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hidden).cuda() * 0.05)
            self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hidden).cuda() * 0.05)
        else:
            self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hidden) * 0.05)
            self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_hidden) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.num_hidden))
                nn.init.uniform_(p, -stdev, stdev)

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.previous_state = self.create_new_state(batch_size)

    def size(self):
        return self.num_inputs, self.num_hidden

    def forward(self, x):
        x = x.unsqueeze(0)
        outp, self.previous_state = self.lstm(x, self.previous_state)
        outp = self.out(outp)
        return outp.squeeze(0), self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
