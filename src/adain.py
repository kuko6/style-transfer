import torch
from torch import nn


def mi(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3])

def sigma(x: torch.Tensor, epsilon=1e-5) -> torch.Tensor:
    return torch.sqrt(torch.sum(((x - mi(x))**2 + epsilon), dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3]))

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return (torch.mul(sigma(style, self.epsilon), ((content - mi(content)) / sigma(content, self.epsilon))) + mi(style))
