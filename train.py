import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

from PIL import Image
import torch.nn.functional as F
from torch.optim import Adam
from utils import forward_diffusion_sample, reverse_process_img_and_plot

from config import *
from networks import SimpleUnet
from utils import linear_beta_schedule

image_dir = r'D:\Datasets\35D Objects\all'  # A raw string literal is prefixed with the letter r. When Python encounters a raw string literal, it treats all backslashes as literal characters, rather than as escape sequences.
image_path = Path(image_dir)
files = list(image_path.glob('*.jpg'))  # will list the entire path


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get all image paths
        image_folder_path = Path(image_dir)
        image_paths = list(image_folder_path.glob('*.jpg'))  # will list the entire path

        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image


img_to_tensor_transform = transforms.Compose([
    transforms.ToTensor(),  # converts PIL or numpy arr to tensor and scales values to [0, 1]
    transforms.Resize((128, 160), antialias=None),  # reduce res for speed
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda a: (a * 2) - 1)  # Scale between [-1, 1]
])

dataset = ImageDataset(root_dir=image_dir, transform=img_to_tensor_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test = next(iter(dataloader))
# test.shape
# image_shape = (240, 320)
image_shape = (128,
               160)  # both dims should be a multiple of 2 for the purpose of clean downsampling. OR have enough 2's in its prime factorization!
# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas,
                               axis=0)  # if input is a vector of size N, the result will also be a vector of size N, where y_i = prod(X_1 x ... x X_i)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # sqrt(a_bar) scaling factor in forward process
sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1. - alphas_cumprod)  # sqrt(1 - a_bar) Standard dev of noise in forward process

alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (
        1. - alphas_cumprod)  # variance used when sampling in the reverse process. ie we add noise to the predicted mean in each reverse step
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

schedules = {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod, "sqrt_alphas_cumprod": sqrt_alphas_cumprod, "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod, "alphas_cumprod_prev": alphas_cumprod_prev, "posterior_variance": posterior_variance, "sqrt_recip_alphas": sqrt_recip_alphas}

model = SimpleUnet()

print("Num params: ", sum(p.numel() for p in model.parameters()))
model


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, schedules, device='cuda')
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)


model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    print(f"epoch: {epoch}")
    for step, X in enumerate(dataloader):
        #         print(X.shape[0], batch_size, X.shape[0] != batch_size)
        if X.shape[0] != batch_size:  # the last batch might contain a different number of images than batch size
            continue
        print(f"step: {step}")
        optimizer.zero_grad()
        t = torch.randint(0, T, (batch_size,), device=device)
        loss = get_loss(model, X, t)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step in [batch_size-2]:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            reverse_process_img_and_plot(model, epoch, image_shape, schedules, device='cuda')
