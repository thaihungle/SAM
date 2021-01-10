import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import json
import logging

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




def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.to("cpu").parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is not None:
                shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

