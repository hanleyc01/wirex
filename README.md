# wirex 

`wirex` is a library and experimental package
which provides experiments with potential Hebbian alternatives to Dense Associative
Memories.

## What's in a name?

`wirex` is a portmanteau of *wire* with [`jax`](https://github.com/jax-ml/jax).
The "wire" comes from the Hebbian adage:
> Neurons that fire together, *wire* together.

## What is a "Hebbian" associative Memory?

What we will be calling a "Hebbian" associative memory is a single
layer associative memory $M$ with a layer of $D$ neurons with 
full lateral connections. The $D \times D$ lateral connections
between the neurons is represented by the weight matrix $W$.

The weight matrix is updated according to the following family of rules:

$W^{(t+1)} = g (W^{(t)}; f(x), x),$

where $g$ is our *Hebbian learning function* and $f$ is our *activation function*.

## Running experiments

The goal of the experimental side of this project is to test the absolute
limits of Hebbian associative memories. To that end, we will be generating
random data (varying interesting and relevant dimensions of the data, e.g. correlation
between generated data)

# Building

This project is built with [`uv`](https://docs.astral.sh/uv/). Make sure you 
have it installed using your system's package manager.

Clone the `wirex` repository:
```sh
$ git clone https://github.com/hanleyc01/wirex.git && cd wirex
```
Sync your project with the `uv.lock` file:
```sh
$ uv sync
```

# Contributing

This project uses the [`ruff`](https://docs.astral.sh/ruff/) formatter with it's default settings.
Install `ruff` using your package manager, and before committing, run:
```sh
$ uv format 
$ uvx ruff check --select I --fix # sort imports
```

Commits use the [Conventional Git](https://www.conventionalcommits.org/en/v1.0.0/) specification
for commits.
