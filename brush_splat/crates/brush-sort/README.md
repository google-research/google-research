WebGPU compatible radix sort. It's based on [this](https://github.com/googlefonts/compute-shader-101/pull/31) implementation, which in turn is based on FidelityFX Radix sort.

It allows sorting up to a given number of bits, and sorting an array with a GPU known number of elements using indirect dispatches.
