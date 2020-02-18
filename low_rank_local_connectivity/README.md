Revisiting Spatial Invariance with Low-Rank Local Connectivity
https://arxiv.org/abs/2002.02959

Work in progress

This is the directory for the low-rank locally connected layer and experiments.

We develop a low-rank locally connected (LRLC) layer that can parametrically 
adjust the degree of spatial invariance. This layer is one particular
method to relax spatial invariance by reducing weight sharing. Rather than 
learning a single filter bank to apply at all positions, as in a convolutional 
layer, or different filter banks, as in a locally connected layer, 
the LRLC layer learns a set of K filter banks, which are linearly combined using
K combining weights per spatial position.
