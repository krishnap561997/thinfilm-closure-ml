import torch
from torch import nn


class WindowMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width=64, hidden_layers=4):
        super().__init__()
        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError('Hidden width and layer count must be positive')
        layers = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(input_dim, hidden_width), nn.Tanh()]
            input_dim = hidden_width
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PhysicalMLP(nn.Module):
    """Accept raw nondimensional inputs; return raw nondimensional targets."""
    def __init__(self, architecture, normalization):
        super().__init__()
        self.mlp = WindowMLP(**architecture)
        for name, values in normalization.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

    def standardized(self, x):
        return self.mlp((x - self.x_mean) / self.x_std)

    def forward(self, x):
        return self.standardized(x) * self.y_std + self.y_mean
