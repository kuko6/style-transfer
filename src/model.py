import torch
from torch import nn
from torchvision.models import vgg19
from torchinfo import summary

from adain import AdaIN


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
        self.encoder = nn.Sequential(*list(vgg19(pretrained=True).features)[:21])

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # set padding in conv layers to reflect
        # create dict for saving activations used in the style loss
        self.activations = {}
        for i, module in enumerate(self.encoder.children()):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = 'reflect'

            if i in [1, 6, 11, 20]:
                module.register_forward_hook(self._save_activations(i))
        
        self.AdaIN = AdaIN()
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.Tanh()
        )

    # https://stackoverflow.com/a/68854535
    def _save_activations(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def forward(self, content, style):
        enc_content = self.encoder(content)
        enc_style = self.encoder(style)
        
        self.t = self.AdaIN(enc_content, enc_style)
        self.t = (1.0 - self.alpha) * enc_content + self.alpha * self.t
        out = self.decoder(self.t)

        return out
    

if __name__ == '__main__':
    print(summary(Model()))