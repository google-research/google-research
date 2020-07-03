#include <iostream>
#include <cassert>
#include "cuda_ops.h"  // NOLINT
#include "cuda_runtime.h"  // NOLINT



__global__ void binary_build_kernel(int n_ints, int n_feats, int* lens,
    uint32_t* bits, float* outptr)
{
    int row = blockIdx.x;
    float* feat_ptr = outptr + row * n_feats;
    uint32_t* cur_bits = bits + row * n_ints;
    int bit_start = threadIdx.x;
    int bit_end = lens[row];
    int bit_steps = blockDim.x;
    for (int i = bit_start; i < bit_end; i += bit_steps)
    {
        int slot = i / 32;
        uint32_t pos = i % 32;
        uint32_t bit = cur_bits[slot] & ((uint32_t)1 << pos);
        feat_ptr[i] = bit ? 1 : -1;
    }
}

void build_binary_mat(int n_rows, int n_ints, int n_feats, int* lens,
                      uint32_t* bits, float* outptr)
{
    int* lens_gpu;
    uint32_t* bits_gpu;
    cudaError_t t = cudaMalloc(&lens_gpu, sizeof(int) * n_rows);
    assert(t == cudaSuccess);
    t = cudaMalloc(&bits_gpu, sizeof(uint32_t) * n_ints * n_rows);
    assert(t == cudaSuccess);

    cudaMemcpy(lens_gpu, lens, sizeof(int) * n_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(bits_gpu, bits, sizeof(uint32_t) * n_rows * n_ints,
               cudaMemcpyHostToDevice);

    dim3 grid(n_rows);
    dim3 block(1024);
    binary_build_kernel<<<grid, block>>>(n_ints, n_feats, lens_gpu,
                                         bits_gpu, outptr);
    cudaFree(lens_gpu);
    cudaFree(bits_gpu);
}
