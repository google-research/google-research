// Copyright 2022 The Google Research Authors.
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

#ifndef TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_UTILS_H_
#define TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_UTILS_H_

#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tf3d {

// Represents `dims` dimensional coordinates.
template <int dims>
struct EIGEN_ALIGN_TO_BOUNDARY(4) Coordinates {
  int32 v[dims];  // The coordinate values.

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Coordinates() = default;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit Coordinates(const int32* src) {
#pragma unroll
    for (int i = 0; i < dims; ++i) {
      v[i] = src[i];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator==(
      const Coordinates& rhs) const {
    for (int i = 0; i < dims; ++i) {
      if (v[i] != rhs[i]) return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Coordinates
  operator+(const Coordinates& rhs) const {
    Coordinates result;
#pragma unroll
    for (int i = 0; i < dims; ++i) {
      result[i] = v[i] + rhs[i];
    }
    return result;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32 operator[](const int i) const {
    return v[i];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32& operator[](const int i) {
    return v[i];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32 hash() const {
    if (dims <= 3) {
      // Optimized Spatial Hashing for Collision Detection of Deformable
      // Objects. by M. Teschner et al. 2003
      //
      // TODO(laigd): measure the performance (collision rate) of the FNV hash
      // function below and merge with it.
      int64_t hash = (static_cast<int64_t>(v[0]) * 73856093) ^
                     (static_cast<int64_t>(v[1]) * 19349663);
      if (dims == 2) return static_cast<int32_t>(hash);
      return static_cast<int32_t>(hash ^
                                  (static_cast<int64_t>(v[2]) * 83492791));
    }

    // FNV Hash function.
    int32 hash = 16777619;
#pragma unroll
    for (auto i : v) {
      hash *= 2166136261;
      hash ^= i;
    }
    return hash;
  }
};

template <int dims>
struct CoordinatesHasher {
  std::size_t operator()(const Coordinates<dims>& coords) const noexcept {
    return coords.hash();
  }
};

// Represents the spatial dimentions of the filter (kernel) in a convolution.
// TODO(laigd): consider inheriting from Coordinates.
template <int dims>
struct FilterSpatialDims : public Coordinates<dims> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE FilterSpatialDims() = default;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit FilterSpatialDims(
      const int32* src)
      : Coordinates<dims>(src) {}

  static Status FromFilterShape(const TensorShape& filter_shape,
                                FilterSpatialDims<dims>* out) {
    for (int i = 0; i < dims; ++i) {
      const int d = filter_shape.dim_size(i);
      if ((d & 1) == 0) {
        return errors::InvalidArgument(
            "Filter spatial dimensions need to be odd numbers for submanifold "
            "convolutions.");
      }
      out->v[i] = d;
    }
    return Status::OK();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int Size() const {
    int result = 1;
#pragma unroll
    for (int i = 0; i < dims; ++i) {
      result *= this->v[i];
    }
    return result;
  }
};

// Iterator for neighbor coordinates. Usage:
//
//   NeighborIterator<dims> iter(...);
//   while (iter.Next()) {
//     auto neighbor = iter.Get();
//     ...
//   }
template <int dims>
struct NeighborIterator {
  // Get the delta of coordinate values of the neighbor given its scalar offset
  // id `neighbor_id`.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Coordinates<dims> GetOffset(
      const FilterSpatialDims<dims>& filter_size, int neighbor_id) {
    Coordinates<dims> result;
#pragma unroll
    for (int i = dims - 1; i >= 0; --i) {
      const int div = neighbor_id / filter_size[i];
      result[i] = (neighbor_id - div * filter_size[i]) - (filter_size[i] >> 1);
      neighbor_id = div;
    }
    return result;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  NeighborIterator(const int32* __restrict__ current_coords,
                   const FilterSpatialDims<dims>& in_filter_size)
      : filter_size(in_filter_size) {
#pragma unroll
    for (int i = 0; i < dims; ++i) {
      beg[i] = current_coords[i] - (in_filter_size[i] >> 1);
    }
#pragma unroll
    for (int i = 0; i < dims - 1; ++i) {
      offset[i] = 0;
    }
    offset[dims - 1] = -1;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Coordinates<dims> Get() const {
    return beg + offset;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool Next() {
    ++offset[dims - 1];
#pragma unroll
    for (int i = dims - 1; i > 0; --i) {
      if (offset[i] == filter_size[i]) {
        offset[i] = 0;
        ++offset[i - 1];
      }
    }
    return offset[0] != filter_size[0];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32 Offset() const {
    int32 result = offset[0];
#pragma unroll
    for (int i = 1; i < dims; ++i) {
      result = result * filter_size[i] + offset[i];
    }
    return result;
  }

 private:
  const FilterSpatialDims<dims>& filter_size;
  Coordinates<dims> beg;
  Coordinates<dims> offset;
};

// Validate the inputs to the convolution.
// - is_grad_op: whether the convolution to run is a gradient op.
// - dims: 2 (for 2D) or 3 (for 3D).
Status ValidateConvInputs(bool is_grad_op, int dims, OpKernelContext* ctx);

}  // namespace tf3d
}  // namespace tensorflow

#endif  // TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_UTILS_H_
