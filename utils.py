import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

from config import T


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):

        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def extract_batch_of_values(vals, t, x_shape):
    """
Returns a specific index t of a passed list of values vals
while considering the batch dimension.

Args:
    vals: tensor. shape=(T,) values in a list extending across all timesteps. This will be on the cpu so we must move t to cpu
    t: tensor. shape=(batch_size,) list of randomly sampled timesteps included in the current batch
    x_shape: shape tuple=(batch_size, channels, H, W) Shape of the minibatch

Returns:
    values at indices t, reshaped to allow multiplication with a batch of images: newshape=(batch_size, 1, 1, 1)
"""
    batch_size = t.shape[0]

    #     out = vals.gather(-1, t.cpu())  #  using the timesteps in t as indices, pull out the corresponding values in vals.
    out = vals[
        t.cpu()]  # values for each timestep in the batch. tensor. shape=(batch_size,) i think this is equiv to using .gather.
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)  # newshape=(batch_size, 1, 1, 1)


def forward_diffusion_sample(x_0, t, schedules, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it at timestep t. Done in BATCH
    Args:
        x_0: tensor. batch of images (batch_size, channels, H, W)
        t: tensor of time steps (batch_size,)
    """
    noise = torch.randn_like(x_0)  # random val from N(0, 1). returns a batch of gaussian noise images
    sqrt_alphas_cumprod_t = extract_batch_of_values(schedules["sqrt_alphas_cumprod"], t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_batch_of_values(
        schedules["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(
        device)  # we return the noise because it will go into the loss function. the model tries to predict the noise to go towards x_{t-1} not x_0


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


@torch.no_grad()
def reverse_process_batch(model, x, t, schedules):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = extract_batch_of_values(schedules["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_batch_of_values(
        schedules["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = extract_batch_of_values(schedules["sqrt_recip_alphas"], t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = extract_batch_of_values(schedules["posterior_variance"], t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def reverse_process_img_and_plot(model, epoch, image_shape, schedules, device='cuda'):
    # Sample noise
    img = torch.randn((1, 3, image_shape[0], image_shape[1]),
                      device=device)  # image with pixels drawn from normal distribution (initial condition)
    plt.figure(figsize=(15*6, 15))
    plt.axis('off')
    num_images = 6  # number of images to plot
    stepsize = int(T / num_images)
    j = 0
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)  # tensor filled with fill value i
        img = reverse_process_batch(model, img, t, schedules)  # shape=(batchsize, 3, H, W)
        img = torch.clip(img, -1.0, 1.0)
        if i % stepsize == 0:
            if 1 <= j <= num_images:
                plt.subplot(1, num_images, j)
                show_tensor_image(img.detach().cpu())
            j += 1
    #     plt.show()
    plt.savefig(f"C:\\Users\\MrLin\\OneDrive\\Documents\\Experiments\\Diffusion Models\\figs\\epoch_{epoch}.png",
                bbox_inches='tight', pad_inches=0, dpi=300)
