import os
import glob
import numpy as np
import wandb
import copy
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchinfo import summary

from utils import StyleContentDataset, DataStore, denorm_img
from loss import Loss
from model import Model


config = {
    "lr": 1e-4,
    "max_iter": 80000,
    "logging_interval": 100,
    "preview_interval": 1000,
    "batch_size": 4,
    "activations": "ReLU",
    "optimizer": "Adam",
    "lambda": 7
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

def prepare_data(style_dir, content_dir, preview_dir):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Training images
    transform = transforms.Compose([transforms.Resize(512), transforms.RandomCrop(256)])
    style_imgs = glob.glob(os.path.join(style_dir, '*.jpg'))
    content_imgs = glob.glob(os.path.join(content_dir, '*.jpg'))

    train_dataset = StyleContentDataset(style_imgs, content_imgs, transform=transform, normalize=norm)
    datastore = DataStore(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Preview images
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])
    preview_style_imgs = glob.glob(os.path.join(preview_dir, 'style/*.jpg'))
    preview_content_imgs = glob.glob(os.path.join(preview_dir, 'content/*.jpg'))

    # preview_dataset = StyleContentDataset(preview_style_imgs, preview_content_imgs, transform=transform, normalize=norm)
    preview_dataset = StyleContentDataset(preview_style_imgs, [preview_content_imgs[8]] * len(preview_style_imgs), transform=transform, normalize=norm)
    preview_datastore = DataStore(preview_dataset, batch_size=len(preview_dataset), shuffle=False)
    
    return datastore, preview_datastore


def preview(model: Model, datastore: DataStore, iteration, save=False, use_wandb=False):
    model.eval()
    with torch.no_grad():
        # np.random.shuffle(datastore.dataset.style_imgs)
        # np.random.shuffle(datastore.dataset.content_imgs)
        
        style, content = datastore.get()
        style, content = style.to(device), content.to(device)
        out = model(content, style)
        
        fig, axs = plt.subplots(8, 6, figsize=(20, 26))
        axs = axs.flatten()
        i = 0
        for (s, c, o) in zip(style, content, out): # style, content, out
            axs[i].imshow(denorm_img(s.cpu()).permute(1, 2, 0))
            axs[i].axis('off')
            axs[i].set_title('style')
            axs[i+1].imshow(denorm_img(c.cpu()).permute(1, 2, 0))
            axs[i+1].axis('off')
            axs[i+1].set_title('content')
            axs[i+2].imshow(denorm_img(o.cpu()).permute(1, 2, 0))
            axs[i+2].axis('off')
            axs[i+2].set_title('output')
            i += 3
         
        if save:
            fig.savefig(f'outputs/{iteration}_preview.png')
            plt.close(fig)
        
        if use_wandb:
            wandb.log({'preview': wandb.Image(f'outputs/{iteration}_preview.png')}, step=iteration)    


def train_one_iter(datastore: DataStore, model: Model, optimizer: torch.optim.Adam, loss_fn: Loss):
    model.train()
    
    style, content = datastore.get()
    style, content = style.to(device), content.to(device)

    # Forward
    out = model(content, style)

    # Save activations
    style_activations = copy.deepcopy(model.activations)
    
    enc_out = model.encoder(out)
    out_activations = model.activations

    # Compute loss
    loss = loss_fn(enc_out, model.t, out_activations, style_activations)

    # Update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_fn.loss_c.item(), loss_fn.loss_s.item()


def train(datastore, preview_datastore, model: Model, optimizer: torch.optim.Adam, use_wandb=False):
    train_history = {'style_loss': [], 'content_loss': [], 'loss': []}
    
    # optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['lr'])
    loss_fn = Loss(lamb=config['lambda'])

    for i in range(config['max_iter']):
        loss, content_loss, style_loss = train_one_iter(datastore, model, optimizer, loss_fn)
        train_history['loss'].append(loss)
        train_history['style_loss'].append(style_loss)
        train_history['content_loss'].append(content_loss)

        if i%config['logging_interval'] == 0:
            print(f'iter: {i}')
            print(f'loss: {loss:>5f}, style loss: {style_loss:>5f}, content loss: {content_loss:>5f}')
            print('-------------------------------')

            if use_wandb:
                wandb.log({
                    'iter': i, 'loss': loss, 'style_loss': style_loss, 'content_loss': content_loss
                })

        if i%config['preview_interval'] == 0:
            torch.save({
                'iter': i, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()
            }, 'outputs/checkpoint.pt')
            preview(model, preview_datastore, i, save=True, use_wandb=use_wandb)

    return train_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, help='path to content dataset')
    parser.add_argument('--style_path', type=str, help='path to content dataset')
    parser.add_argument('--preview_path', type=str, help='path to preview dataset')
    parser.add_argument('--wandb', type=str, help='wandb id')
    parser.add_argument('--model_path', type=str, help='path to model')
    args = parser.parse_args()

    use_wandb = False
    wandb_key = args.wandb
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="assignment-3", name="", reinit=True, config=config)    
        use_wandb = True

    if args.content_path and args.style_path and args.preview_path:
        content_dir = args.content_path
        style_dir = args.style_path
        preview_dir = args.preview_path
    else:
        print('You didnt specify the data path >:(')
        return
    
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    datastore, preview_datastore = prepare_data(style_dir, content_dir, preview_dir)
    
    model = Model()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['lr'])
    if args.model_path:
        # From checkpoint
        checkpoint = torch.load('outputs/checkpoint.pt')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        config['max_iter'] -= checkpoint['iter']

        # From final model
        # model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    # print(summary(model))
    model.to(device)
    
    train(datastore, preview_datastore, model, optimizer, use_wandb)
    
    torch.save(model.state_dict(), 'outputs/model.pt')
    if use_wandb:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('outputs/model.pt')
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    main() 