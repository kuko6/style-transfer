import torch
from torch import nn
import torch.nn.functional as F

from adain import mi, sigma


class Loss(nn.Module):
    def __init__(self, lamb=8):
        super().__init__()
        self.lamb = lamb

    def content_loss(self, enc_out: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(enc_out, t)
    
    def style_loss(self, out_activations: dict, style_activations: dict) -> torch.Tensor:
        means, sds = 0, 0
        for out_act, style_act in zip(out_activations.values(), style_activations.values()):
            means += F.mse_loss(mi(out_act), mi(style_act))
            sds += F.mse_loss(sigma(out_act), sigma(style_act))
            
        return means + sds

    def forward(self, enc_out: torch.Tensor, t: torch.Tensor, out_activations: dict, style_activations: dict) -> torch.Tensor:
        # batch_size = enc_out.shape[0]
        # self.loss_c = self.content_loss(enc_out, t) / batch_size
        # self.loss_s = self.style_loss(out_activations, style_activations) / batch_size

        self.loss_c = self.content_loss(enc_out, t)
        self.loss_s = self.style_loss(out_activations, style_activations)
        
        return (self.loss_c + self.lamb * self.loss_s)