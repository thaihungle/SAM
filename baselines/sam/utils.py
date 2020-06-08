import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class MLP(nn.Module):
    def __init__(self, equation, in_features, hidden_size, out_size):
        super(MLP, self).__init__()
        self.equation = equation
        # Layers
        # 1
        self.W1 = nn.Parameter(torch.zeros(in_features, hidden_size))
        nn.init.xavier_uniform_(self.W1.data)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        # 2
        self.W2 = nn.Parameter(torch.zeros(hidden_size, out_size))
        nn.init.xavier_uniform_(self.W2.data)
        self.b2 = nn.Parameter(torch.zeros(out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum(self.equation, x, self.W1) + self.b1)
        out = torch.tanh(torch.einsum(self.equation, hidden, self.W2) + self.b2)
        return out


class OptionalLayer(nn.Module):
    def __init__(self, layer: nn.Module, active: bool = False):
        super(OptionalLayer, self).__init__()
        self.layer = layer
        self.active = active
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active:
            return self.layer(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        normalized = (x - mu) / (torch.sqrt(sigma + self.eps))
        return normalized * self.gain + self.bias


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, multiplier: float, steps: int):
        self.multiplier = multiplier
        self.steps = steps
        super(WarmupScheduler, self).__init__(optimizer=optimizer)

    def get_lr(self):
        if self.last_epoch < self.steps:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return self.base_lrs

    def decay_lr(self, decay_factor: float):
        self.base_lrs = [decay_factor * base_lr for base_lr in self.base_lrs]
