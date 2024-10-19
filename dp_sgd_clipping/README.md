## Python implementation of fast and memory-efficient DP-SGD clipping

This directory contains a Python implementation of fast and memory-efficient
DP-SGD clipping.

## Overview

The implementation is based on the paper "Efficient Gradient Clipping Methods in
DP-SGD for Convolution Models" by W. Kong, M. Ribero, and G. Velegkas.

The paper provides three fast and memory-efficient algorithm for DP-SGD clipping
for convolutional neural networks.

## Notes
The 'fast_cnn_grad_norm.py' file contains the implementation of all the three
algorithms for fast and memory efficient norm computation that can be used as
part of the DP-SGD algorithm.

The 'in_place_fast_grad_norm' function implements the direct norm computation
algorithm in an in-place manner.

The 'in_place_ghost_norm' function implements the ghost norm clipping algorithm
in an in-place manner.

The 'in_place_norm_fft' function implements the norm computation algorithm that
is based on FFT.

The 'naive_fast_grad_norm' function implements the direct norm computation
algorithm in a non-in-place manner, as proposed by
https://arxiv.org/pdf/2205.10683.

The 'naive_ghost_norm' function implements the ghost norm computation
algorithm in a non-in-place manner, as proposed by
https://arxiv.org/pdf/2205.10683.

Since all these algorithms are incomparable in terms of runtime efficiency, we
have implemented the function 'grad_norm_computation_selector' that selects the
algorithm based on the underlying parameters.
