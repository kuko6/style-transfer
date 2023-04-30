import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import numpy as np


def denorm_img(img: torch.Tensor) -> torch.Tensor:
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    return torch.clip(img * std + mean, min=0, max=1)


class StyleContentDataset(Dataset):
    def __init__(self, style_imgs, content_imgs, transform=None, normalize=None):
        self.style_imgs = style_imgs
        self.content_imgs = content_imgs
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        if len(self.style_imgs) < len(self.content_imgs):
            return len(self.style_imgs)
        else:
            return len(self.content_imgs)
    
    def __getitem__(self, idx):
        try:
            style = read_image(self.style_imgs[idx], ImageReadMode.RGB).float() / 255.0
            content = read_image(self.content_imgs[idx], ImageReadMode.RGB).float() / 255.0
        except RuntimeError:
            print(self.style_imgs[idx])
            print(self.content_imgs[idx])
            style = read_image(self.style_imgs[0], ImageReadMode.RGB).float() / 255.0
            content = read_image(self.content_imgs[0], ImageReadMode.RGB).float() / 255.0

        if self.normalize:
            style = self.normalize(style)
            content = self.normalize(content)
        
        if self.transform:
            style = self.transform(style)
            content = self.transform(content)
        
        return style, content
    

class DataStore():
    def __init__(self, dataset: StyleContentDataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        self.iterator = iter(self.dataloader)

    def get(self):
        try:
           style, content = next(self.iterator)
        except (StopIteration):
            # print('| Repeating |')
            # np.random.shuffle(self.dataset.style_imgs)
            self.iterator = iter(self.dataloader)
            style, content = next(self.iterator)
        
        return style, content