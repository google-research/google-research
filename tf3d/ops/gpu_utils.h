// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TF3D_OPS_GPU_UTILS_H_
#define TF3D_OPS_GPU_UTILS_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "absl/base/integral_types.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tf3d {
namespace cuda {

template <typename T>
__host__ __device__ __forceinline__ T Square(T a) {
  return a * a;
}

template <typename T>
__host__ __device__ __forceinline__ T FillLowerBits(T n);

template <>
__host__ __device__ __forceinline__ uint8 FillLowerBits<uint8>(uint8 n) {
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  return n;
}

template <>
__host__ __device__ __forceinline__ uint16 FillLowerBits<uint16>(uint16 n) {
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  return n;
}

template <>
__host__ __device__ __forceinline__ uint32 FillLowerBits<uint32>(uint32 n) {
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n;
}

template <>
__host__ __device__ __forceinline__ unsigned long long FillLowerBits<unsigned long long>(unsigned long long n) {
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n;
}

// Return m which is a power of two such that n<=m<2n.
// Requires: 1<=n
template <typename T>
__host__ __device__ __forceinline__ T NextPowerOfTwoGE(T n) {
  return FillLowerBits<T>(n - 1) + 1;
}

// Return m which is a power of two such that n<m<=2n.
// Requires: 1<=n
template <typename T>
__host__ __device__ __forceinline__ T NextPowerOfTwoGT(T n) {
  return FillLowerBits<T>(n) + 1;
}

// Return m which is a power of two such that n/2<m<=n.
// Requires: 1<n
template <typename T>
__host__ __device__ __forceinline__ T PreviousPowerOfTwoLE(T n) {
  n = FillLowerBits<T>(n);
  return (n + 1) >> 1;
}

// Return m which is a power of two such that n/2<=m<n.
// Requires: 1<n
template <typename T>
__host__ __device__ __forceinline__ T PreviousPowerOfTwoLT(T n) {
  n = FillLowerBits<T>(n - 1);
  return (n + 1) >> 1;
}

}  // namespace cuda
}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TF3D_OPS_GPU_UTILS_H_
