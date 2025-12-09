"""Experimentation with wirex."""

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from wirex.models.rate.apr import ActivityProductRule
from wirex.models.rate.hopfield import Hopfield

matplotlib.use("Qt5Agg")

DATA_CACHE = "./.cache/mnist/"
pxw, pxh = 28, 28


def transform(data: torch.Tensor) -> NDArray[np.float32]:
    dataarr: NDArray[np.float32] = np.array(data, dtype=jnp.float32)
    dataarr = rearrange(dataarr, "w h -> (w h)")
    dataarr[dataarr > 0.0] = 1.0
    dataarr[dataarr == 0.0] = -1.0
    return dataarr


def show_im(im: torch.Tensor | jax.Array, title: str = "") -> None:
    im = rearrange(im, "(w h) -> w h", w=pxw, h=pxh)
    plt.imshow(im)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def main() -> None:
    mnist_train = MNIST(DATA_CACHE, train=True, transform=transform, download=True)
    mnist_data_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
    mnist_it = iter(mnist_data_loader)
    mnist_data, _ = next(mnist_it)

    Xi = jnp.array(mnist_data[:2], dtype=jnp.float32)
    hopfield = Hopfield(Xi.T @ Xi)
    query = Xi[1]
    result = hopfield(query)
    show_im(result, "Hopfield_result")
    plt.show()

    apr = ActivityProductRule.init(Xi.shape[-1], 1).fit(Xi)
    result = apr(query)
    show_im(result, "APR Result")
    plt.show()


if __name__ == "__main__":
    main()
