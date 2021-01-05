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

#ifndef SCANN__UTILS_INTRINSICS_AVX1_H_
#define SCANN__UTILS_INTRINSICS_AVX1_H_

#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/sse4.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <x86intrin.h>

namespace tensorflow {
namespace scann_ops {

class M256_8Xfloat {
 public:
  using ClassName = M256_8Xfloat;

  SCANN_INLINE static constexpr size_t BlockSize() { return 8; }

  SCANN_AVX1_INLINE M256_8Xfloat() {}

  SCANN_AVX1_INLINE M256_8Xfloat(__m256 val) : val_(val) {}

  SCANN_AVX1_INLINE static ClassName Zeros() { return {_mm256_setzero_ps()}; }

  SCANN_AVX1_INLINE static ClassName Broadcast(float x) {
    return {_mm256_set1_ps(x)};
  }

  SCANN_AVX1_INLINE __m256 val() const { return val_; }

  SCANN_AVX1_INLINE static ClassName Load(const float* address) {
    return {_mm256_loadu_ps(address)};
  }

  SCANN_AVX1_INLINE void Store(float* address) const {
    _mm256_storeu_ps(address, val_);
  }

  SCANN_AVX1_INLINE float GetLowElement() const {
    return _mm_cvtss_f32(_mm256_castps256_ps128(val_));
  }

  SCANN_AVX1_INLINE ClassName operator+(const ClassName& other) const {
    return {_mm256_add_ps(val_, other.val_)};
  }

  SCANN_AVX1_INLINE ClassName operator-(const ClassName& other) const {
    return {_mm256_sub_ps(val_, other.val_)};
  }

  SCANN_AVX1_INLINE ClassName operator*(const ClassName& other) const {
    return {_mm256_mul_ps(val_, other.val_)};
  }

  SCANN_AVX1_INLINE ClassName operator&(const ClassName& other) const {
    return {_mm256_and_ps(val_, other.val_)};
  }

  SCANN_AVX1_INLINE ClassName operator|(const ClassName& other) const {
    return {_mm256_or_ps(val_, other.val_)};
  }

  SCANN_AVX1_INLINE ClassName operator<(const ClassName& other) const {
    return {_mm256_cmp_ps(val_, other.val_, _CMP_LT_OS)};
  }

  SCANN_AVX1_INLINE ClassName operator<=(const ClassName& other) const {
    return {_mm256_cmp_ps(val_, other.val_, _CMP_LE_OS)};
  }

  SCANN_AVX1_INLINE ClassName operator==(const ClassName& other) const {
    return {_mm256_cmp_ps(val_, other.val_, _CMP_EQ_OS)};
  }

  SCANN_AVX1_INLINE ClassName operator>=(const ClassName& other) const {
    return {_mm256_cmp_ps(val_, other.val_, _CMP_GE_OS)};
  }

  SCANN_AVX1_INLINE ClassName operator>(const ClassName& other) const {
    return {_mm256_cmp_ps(val_, other.val_, _CMP_GT_OS)};
  }

  SCANN_AVX1_INLINE int MaskFromHighBits() const {
    return _mm256_movemask_ps(val_);
  }
  SCANN_AVX2_INLINE int MaskFromHighBitsAvx2() const {
    return _mm256_movemask_ps(val_);
  }

  __m256 val_;
};

SCANN_AVX2_INLINE uint32_t MaskFromHighBits(M256_8Xfloat v0, M256_8Xfloat v1,
                                            M256_8Xfloat v2, M256_8Xfloat v3) {
  const uint32_t m00 = v0.MaskFromHighBits();
  const uint32_t m08 = v1.MaskFromHighBits();
  const uint32_t m16 = v2.MaskFromHighBits();
  const uint32_t m24 = v3.MaskFromHighBits();
  return m00 + (m08 << 8) + (m16 << 16) + (m24 << 24);
}

SCANN_AVX2_INLINE uint32_t MaskFromHighBits(M256_8Xfloat v[4]) {
  return MaskFromHighBits(v[0], v[1], v[2], v[3]);
}

namespace avx1 {

template <typename T>
struct SimdImpl;

template <>
struct SimdImpl<int8_t> {
  using type = M128_16Xint8;
};

template <>
struct SimdImpl<uint8_t> {
  using type = M128_16Xuint8;
};

template <>
struct SimdImpl<int16_t> {
  using type = M128_8Xint16;
};

template <>
struct SimdImpl<uint16_t> {
  using type = M128_8Xuint16;
};

template <>
struct SimdImpl<int32_t> {
  using type = M128_4Xint32;
};

template <>
struct SimdImpl<uint32_t> {
  using type = M128_4Xuint32;
};

template <>
struct SimdImpl<float> {
  using type = M256_8Xfloat;
};

template <typename T>
using Simd = typename SimdImpl<T>::type;

}  // namespace avx1
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
