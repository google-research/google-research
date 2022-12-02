// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <THC/THC.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <vector>
#include "stacked_fc_cuda.cuh"


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_LONG(x) TORCH_CHECK(x.type().scalarType() == torch::kInt64, #x " must be a Long tensor")
#define N_THREADS 1024

template <typename scalar_t>
__global__ void stacked_fc_cuda_forward_kernel(
    const scalar_t*  __restrict__ input_feat,
    const long*  __restrict__ input_idx,
    const scalar_t*  __restrict__ weights,
    const scalar_t*  __restrict__ bias,
    scalar_t*  __restrict__ out,
    const int batch_size,
    const int k_nerflets,
    const int n_nerflets,
    const int n_in_channels,
    const int n_out_channels) {

  int b = blockIdx.x;
  int c_in = threadIdx.x;
  int c_out = blockIdx.z;
  int k_idx = blockIdx.y;
  int k_val = input_idx[b * k_nerflets + k_idx];

  int out_idx = b * k_nerflets * n_out_channels + k_idx * n_out_channels + c_out;
  const scalar_t* kth_weights_mat = weights + k_val * n_in_channels * n_out_channels;

  extern __shared__ int data_buffer[];  // array for reduction;
  auto thread_data = reinterpret_cast<scalar_t*>(data_buffer);
  thread_data[c_in] = input_feat[b * n_in_channels + c_in] * kth_weights_mat[c_in * n_out_channels + c_out];
  __syncthreads();

  for (unsigned int s = n_in_channels / 2; s > 0; s >>= 1) {
      if (c_in < s)
          thread_data[c_in] += thread_data[c_in + s];
      __syncthreads();
  }

  if (c_in == 0)
      out[out_idx] = bias[k_val * n_out_channels + c_out] + thread_data[0];
}


torch::Tensor stacked_fc_cuda_forward(
    torch::Tensor input_feat,       // (B, Cin)
    torch::Tensor input_idx,        // (B, K)
    torch::Tensor weights,          // (N, Cin, Cout)
    torch::Tensor bias) {           // (N, 1, Cout)

  CHECK_INPUT(input_feat);
  CHECK_INPUT(input_idx);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_LONG(input_idx);

  auto batch_size = input_idx.size(0);
  auto k_nerflets = input_idx.size(1);
  auto n_nerflets = weights.size(0);
  auto n_in_channels = weights.size(1);
  auto n_out_channels = weights.size(2);

  auto out = torch::empty({batch_size, k_nerflets, n_out_channels}, weights.options());
  const int threads = n_in_channels;    // TODO: optimize
  const dim3 blocks(batch_size, k_nerflets, n_out_channels);

  AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "stacked_fc_cuda_forward", ([&] {
    stacked_fc_cuda_forward_kernel<scalar_t><<<blocks, threads, sizeof(scalar_t) * threads>>>(
        input_feat.data_ptr<scalar_t>(),
        input_idx.data_ptr<long>(),
        weights.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        batch_size, k_nerflets, n_nerflets, n_in_channels, n_out_channels);
  }));

  THCudaCheck(cudaGetLastError());

  return out;
}


template <typename scalar_t>
__global__ void stacked_fc_cuda_backward_kernel_feat(
    const scalar_t*  __restrict__ out,           // (B, K, Cout)
    const scalar_t*  __restrict__ input_feat,    // (B, Cin)
    const long*  __restrict__ input_idx,         // (B, K)
    const scalar_t*  __restrict__ weights,       // (N, Cin, Cout)
    const scalar_t*  __restrict__ grad_out,      // (B, K, Cout)
    scalar_t*  __restrict__ d_feat,              // (B, Cin)
    const int batch_size,
    const int k_nerflets,
    const int n_nerflets,
    const int n_in_channels,
    const int n_out_channels) {

  int b = blockIdx.x;
  int c_out = threadIdx.x;
  int c_in = blockIdx.z;
  int k_idx = blockIdx.y;
  int k_val = input_idx[b * k_nerflets + k_idx];

  extern __shared__ int data_buffer[];  // array for reduction;
  auto thread_data = reinterpret_cast<scalar_t*>(data_buffer);
  thread_data[c_out] = grad_out[b * k_nerflets * n_out_channels + k_idx * n_out_channels + c_out]
                     * weights[k_val * n_in_channels * n_out_channels + c_in * n_out_channels + c_out];
  __syncthreads();

  if (c_out == 0) {
    scalar_t result = 0;
    for (int i = 0; i < n_out_channels; i++)
      result += thread_data[i];
    atomicAdd(d_feat + b * n_in_channels + c_in, result);
  }
}

template <typename scalar_t>
__global__ void stacked_fc_cuda_backward_kernel_w(
    const scalar_t*  __restrict__ input_feat,    // (B, Cin)
    const long*  __restrict__ input_idx,         // (B, K)
    const scalar_t*  __restrict__ grad_out,      // (B, K, Cout)
    scalar_t*  __restrict__ d_w,                 // (N, Cin, Cout)
    const int batch_size,
    const int k_nerflets,
    const int n_nerflets,
    const int n_in_channels,
    const int n_out_channels) {

  int b = blockIdx.x;
  int c_out = threadIdx.x;
  int c_in = blockIdx.z;
  int k_idx = blockIdx.y;
  int k_val = input_idx[b * k_nerflets + k_idx];

  scalar_t result = input_feat[b * n_in_channels + c_in]
                  * grad_out[b * k_nerflets * n_out_channels + k_idx * n_out_channels + c_out];

  atomicAdd(d_w + k_val * n_in_channels * n_out_channels + c_in * n_out_channels + c_out, result);
}

template <typename scalar_t>
__global__ void stacked_fc_cuda_backward_kernel_b(
    const long*  __restrict__ input_idx,         // (B, K)
    const scalar_t*  __restrict__ grad_out,      // (B, K, Cout)
    scalar_t*  __restrict__ d_b,                 // (N, 1, Cout)
    const int batch_size,
    const int k_nerflets,
    const int n_nerflets,
    const int n_in_channels,
    const int n_out_channels) {

  int b = blockIdx.x;
  int c_out = threadIdx.x;
  int k_idx = blockIdx.y;
  int k_val = input_idx[b * k_nerflets + k_idx];

  atomicAdd(d_b + k_val * n_out_channels + c_out, grad_out[b * k_nerflets * n_out_channels + k_idx * n_out_channels + c_out]);
}

std::vector<torch::Tensor> stacked_fc_cuda_backward(
   torch::Tensor out,              // (B, K, Cout)
   torch::Tensor input_feat,       // (B, Cin)
   torch::Tensor input_idx,        // (B, K)
   torch::Tensor weights,          // (B, Cin, Cout)
   torch::Tensor bias,             // (B, 1, Cout)
   torch::Tensor grad_out) {       // (B, K, Cout)

  CHECK_INPUT(out);
  CHECK_INPUT(input_feat);
  CHECK_INPUT(input_idx);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(grad_out);

  auto batch_size = input_idx.size(0);
  auto k_nerflets = input_idx.size(1);
  auto n_nerflets = weights.size(0);
  auto n_in_channels = weights.size(1);
  auto n_out_channels = weights.size(2);

  auto d_feat = torch::zeros_like(input_feat);
  auto d_idx = torch::zeros_like(input_idx);
  auto d_w = torch::zeros_like(weights);
  auto d_b = torch::zeros_like(bias);

  // d_feat
  {
    const int threads_feat = n_out_channels;
    const dim3 blocks_feat(batch_size, k_nerflets, n_in_channels);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "stacked_fc_cuda_backward_feat", ([&] {
      stacked_fc_cuda_backward_kernel_feat<scalar_t><<<blocks_feat, threads_feat, sizeof(scalar_t) * threads_feat>>>(
          out.contiguous().data_ptr<scalar_t>(),
          input_feat.contiguous().data_ptr<scalar_t>(),
          input_idx.contiguous().data_ptr<long>(),
          weights.contiguous().data_ptr<scalar_t>(),
          grad_out.contiguous().data_ptr<scalar_t>(),
          d_feat.contiguous().data_ptr<scalar_t>(),
          batch_size, k_nerflets, n_nerflets, n_in_channels, n_out_channels);
    }));

    THCudaCheck(cudaGetLastError());
  }

  // d_w
  {
    const int threads_w = n_out_channels;
    const dim3 blocks_w(batch_size, k_nerflets, n_in_channels);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "stacked_fc_cuda_backward_w", ([&] {
      stacked_fc_cuda_backward_kernel_w<scalar_t><<<blocks_w, threads_w>>>(
          input_feat.contiguous().data_ptr<scalar_t>(),
          input_idx.contiguous().data_ptr<long>(),
          grad_out.contiguous().data_ptr<scalar_t>(),
          d_w.contiguous().data_ptr<scalar_t>(),
          batch_size, k_nerflets, n_nerflets, n_in_channels, n_out_channels);
    }));

    THCudaCheck(cudaGetLastError());
  }

  // d_b
  {
    const int threads_b = n_out_channels;
    const dim3 blocks_b(batch_size, k_nerflets, 1);

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "stacked_fc_cuda_backward_b", ([&] {
      stacked_fc_cuda_backward_kernel_b<scalar_t><<<blocks_b, threads_b>>>(
          input_idx.contiguous().data_ptr<long>(),
          grad_out.contiguous().data_ptr<scalar_t>(),
          d_b.contiguous().data_ptr<scalar_t>(),
          batch_size, k_nerflets, n_nerflets, n_in_channels, n_out_channels);
    }));

    THCudaCheck(cudaGetLastError());
  }

  return {d_feat, d_idx, d_w, d_b};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stacked_fc_cuda_forward, "Stacked FC forward (CUDA)");
  m.def("backward", &stacked_fc_cuda_backward, "Stacked FC backward (CUDA)");
}
