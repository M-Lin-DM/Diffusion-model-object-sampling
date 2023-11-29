from utils import show_images, forward_diffusion_sample, show_tensor_image
import matplotlib.pyplot as plt
import torch

from config import *

show_images(dataset)

image = next(iter(dataloader))[0]

plt.figure(figsize=(20, 20))
plt.axis('off')
num_images = 5
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)