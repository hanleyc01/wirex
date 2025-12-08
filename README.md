# wirex 

wirex is a library and experimental package
which provides experiments with potential Hebbian alternatives to Dense Associative
Memories.

# What is a "Hebbian" associative Memory?

What we will be calling a "Hebbian" associative memory is a single
layer associative memory $M$ with a layer of $D$ neurons with 
full lateral connections. The $D \times D$ lateral connections
between the neurons is represented by the weight matrix $W$.

The weight matrix is updated according to the following family of rules:
$$
W^{(t+1)} = g (W^{(t)}; f(x), x),
$$
where $g$ is our *Hebbian learning function* and $f$ is our *activation function*.

# Running experiments

The goal of the experimental side of this project is to test the absolute
limits of Hebbian associative memories. To that end, we will be generating
random data (varying interesting and relevant dimensions of the data, e.g. correlation
between generated data)


# Packages used in the project

Because the study will be performance-intensive, we will be using the `jax` library
backend for describing our models.

