"""Developmental testing of the wirex program."""

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from .models.hopfield import Hopfield

matplotlib.use("Qt5Agg")

DATA_CACHE = "./cache/mnist/"
pxw, pxh = 28, 28


def transform(data):
    data = np.array(data, dtype=jnp.float32)
    data = rearrange(data, "w h -> (w h)")
    data[data > 0.0] = 1.0
    data[data == 0.0] = -1.0
    return data


mnist_train = MNIST(DATA_CACHE, train=True, transform=transform, download=True)
mnist_data_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
mnist_it = iter(mnist_data_loader)
mnist_data, _ = next(mnist_it)


def show_im(im, title=""):
    im = rearrange(im, "(w h) -> w h", w=pxw, h=pxh)
    axesim = plt.imshow(im)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return axesim


# show_im(mnist_data[0], title="Testing whether the transform is correct")
# plt.show()

Xi = jnp.array(mnist_data[:2], dtype=jnp.float32)
hopfield = Hopfield(Xi.T @ Xi)
query = Xi[1]
result = hopfield.recall(query)
show_im(result)
plt.show()
