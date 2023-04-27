import torch
from torch import nn
import torch.nn.functional as F

from adain import mi, sigma


class Loss(nn.Module):
    def __init__(self, lamb=8):
        super().__init__()
        self.lamb = lamb

    def content_loss(self, enc_out, t):
        # return torch.linalg.norm(enc_out - t)
        return F.mse_loss(enc_out, t)
    
    def style_loss(self, out_activations, style_activations):
        means, sds = 0, 0
        for out_act, style_act in zip(out_activations.values(), style_activations.values()):
            # means = torch.linalg.norm(mi(out_act) - mi(style_act))
            # sds = torch.linalg.norm(sigma(out_act) - sigma(style_act))    
            means += F.mse_loss(mi(out_act), mi(style_act))
            sds += F.mse_loss(sigma(out_act), sigma(style_act))
            
        return means + sds

    def forward(self, enc_out, t, out_activations, style_activations):
        self.loss_c = self.content_loss(enc_out, t) / t.shape[0]
        self.loss_s = self.style_loss(out_activations, style_activations) / t.shape[0]
        
        return (self.loss_c + self.lamb * self.loss_s)