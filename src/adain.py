import torch
from torch import nn


def mi(x):
    return torch.sum(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3])

def sigma(x, epsilon=1e-5):
    return torch.sqrt(torch.sum(((x - mi(x))**2 + epsilon), dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3]))

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
    
    # def mi(self, x):
    #     return torch.sum(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3])

    # def sigma(self, x):
    #     return torch.sqrt(torch.sum(((x - self.mi(x))**2 + self.epsilon), dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3]))

    def forward(self, content, style):
        # return (torch.mul(self.sigma(style), ((content - self.mi(content)) / self.sigma(content))) + self.mi(style))
        return (torch.mul(sigma(style, self.epsilon), ((content - mi(content)) / sigma(content, self.epsilon))) + mi(style))
