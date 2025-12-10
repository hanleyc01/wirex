# Experiments

The goal of this part of the project is to determine the capacity
scaling laws of Hebbian associative memories *in general*.

To test this, we will be randomly generating Hebbian associative
memories, and then training them on randomly generated data. 

The important factors of variation are as follows:

1. Properties of the random data generated;
2. The values of the Taylor coefficients for the Hebbian learning rule;
3. The number of the Taylor coefficients for the Hebbian learning rule;
4. The dimensionality of the networks generated (derivative of the dimensionality of the data);
5. The number of patterns that the networks are required to store.

The goal of the observations is to come up with a function $f$ which depends 
on (1-5) and gives us the number $c$ of the maximum number of patterns which the
network can reliably retrieve (within some margin of error).

# Preliminary predictions

1. The more correlated that the data is, the less the capacity (see, the idea of "cross-talk");
2. The values of the Taylor coefficients will depend entirely upon being fit to the data at hand;
3. The number of Taylor coefficients in the Hebbian learning rule expansion will be very meaningful at low values,
   with diminishing returns with an increased number of coefficients.
4. The blessing of dimensionality: higher dimensionality lends itself to more pseudo-orthogonal data, and therefore a higher capacity
   (see Hopfield results: roughly 14% of the neurons).
5. Performance should be stable and not degrade until a critical capacity is reached, and then the results should fall off a cliff.
