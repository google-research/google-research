// Copyright 2020 The Google Research Authors.
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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__UTILS_INTRINSICS_SSE4_H_
#define SCANN__UTILS_INTRINSICS_SSE4_H_

#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <x86intrin.h>

namespace tensorflow {
namespace scann_ops {

template <typename ClassName, typename CType>
class M128Mixin {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(M128Mixin);

  SCANN_INLINE static constexpr size_t BlockSize() {
    return 16 / sizeof(CType);
  }

  SCANN_SSE4_INLINE M128Mixin() {}

  SCANN_SSE4_INLINE M128Mixin(__m128i val) : val_(val) {}

  SCANN_SSE4_INLINE static ClassName Zeros() { return {_mm_setzero_si128()}; }

  SCANN_SSE4_INLINE static ClassName Broadcast(const CType& x) {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm_set1_epi8(x)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_set1_epi16(x)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_set1_epi32(x)};
    }
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit broadcast instruction");
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE __m128i val() const { return val_; }

  SCANN_SSE4_INLINE static ClassName Load(const CType* address) {
    return ClassName(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(address)));
  }

  SCANN_SSE4_INLINE void Store(CType* address) const {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(address), val_);
  }

  SCANN_SSE4_INLINE ClassName operator+(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm_add_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_add_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_add_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm_add_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator-(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm_sub_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_sub_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_sub_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm_sub_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator>>(int count) const {
    static_assert(!IsSameAny<CType, int8_t, uint8_t>(),
                  "There is no 8-bit '>>' instruction");
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit '>>' instruction");

    if (IsSame<CType, int16_t>()) {
      return {_mm_srai_epi16(val_, count)};
    }
    if (IsSame<CType, int32_t>()) {
      return {_mm_srai_epi32(val_, count)};
    }

    if (IsSameAny<CType, uint16_t>()) {
      return {_mm_srli_epi16(val_, count)};
    }
    if (IsSame<CType, uint32_t>()) {
      return {_mm_srli_epi32(val_, count)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator<<(int count) const {
    static_assert(!IsSameAny<CType, int8_t, uint8_t>(),
                  "There is no 8-bit '<<' instruction");
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit '<<' instruction");

    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_slli_epi16(val_, count)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_slli_epi32(val_, count)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator==(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm_cmpeq_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_cmpeq_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_cmpeq_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm_cmpeq_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator>(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm_cmpgt_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm_cmpgt_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm_cmpgt_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm_cmpgt_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE ClassName operator<(const ClassName& other) const {
    return (other > *reinterpret_cast<const ClassName*>(this));
  }

  SCANN_SSE4_INLINE ClassName operator&(ClassName other) const {
    return {val_ & other.val_};
  }

  SCANN_SSE4_INLINE ClassName operator|(ClassName other) const {
    return {val_ | other.val_};
  }

  SCANN_SSE4_INLINE ClassName& operator+=(const ClassName& other) {
    val_ = (*this + other).val();
    return *reinterpret_cast<ClassName*>(this);
  }

  SCANN_SSE4_INLINE ClassName& operator-=(const ClassName& other) {
    val_ = (*this - other).val();
    return *reinterpret_cast<ClassName*>(this);
  }

  SCANN_SSE4_INLINE ClassName& operator>>=(int count) {
    val_ = (*this >> count).val();
    return *reinterpret_cast<ClassName*>(this);
  }

 private:
  __m128i val_;
  friend ClassName;
};

class M128_4Xfloat {
 public:
  using ClassName = M128_4Xfloat;

  SCANN_INLINE static constexpr size_t BlockSize() { return 4; }

  SCANN_SSE4_INLINE M128_4Xfloat() {}

  SCANN_SSE4_INLINE M128_4Xfloat(__m128 val) : val_(val) {}

  SCANN_SSE4_INLINE static ClassName Zeros() { return {_mm_setzero_ps()}; }

  SCANN_SSE4_INLINE static ClassName Broadcast(float x) {
    return {_mm_set1_ps(x)};
  }

  SCANN_SSE4_INLINE __m128 val() const { return val_; }

  SCANN_SSE4_INLINE static ClassName Load(const float* address) {
    return {_mm_loadu_ps(address)};
  }

  SCANN_SSE4_INLINE void Store(float* address) const {
    _mm_storeu_ps(address, val_);
  }

  SCANN_SSE4_INLINE float GetLowElement() const { return _mm_cvtss_f32(val_); }

  SCANN_SSE4_INLINE ClassName operator+(const ClassName& other) const {
    return {val_ + other.val_};
  }

  SCANN_SSE4_INLINE ClassName operator-(const ClassName& other) const {
    return {val_ - other.val_};
  }

  SCANN_SSE4_INLINE ClassName operator*(const ClassName& other) const {
    return {val_ * other.val_};
  }

  SCANN_SSE4_INLINE ClassName operator&(const ClassName& other) const {
    return {_mm_and_ps(val_, other.val_)};
  }

  SCANN_SSE4_INLINE ClassName operator|(const ClassName& other) const {
    return {_mm_or_ps(val_, other.val_)};
  }

  SCANN_SSE4_INLINE ClassName operator<(const ClassName& other) const {
    return _mm_cmplt_ps(val_, other.val_);
  }

  SCANN_SSE4_INLINE ClassName operator<=(const ClassName& other) const {
    return _mm_cmple_ps(val_, other.val_);
  }

  SCANN_SSE4_INLINE ClassName operator==(const ClassName& other) const {
    return _mm_cmpeq_ps(val_, other.val_);
  }

  SCANN_SSE4_INLINE ClassName operator>=(const ClassName& other) const {
    return _mm_cmpge_ps(val_, other.val_);
  }

  SCANN_SSE4_INLINE ClassName operator>(const ClassName& other) const {
    return _mm_cmpgt_ps(val_, other.val_);
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    return _mm_movemask_ps(val_);
  }

 private:
  __m128 val_;
};

class M128_4Xint32 : public M128Mixin<M128_4Xint32, int32_t> {
 public:
  using M128Mixin = M128Mixin<M128_4Xint32, int32_t>;

  using M128Mixin::M128Mixin;

  SCANN_SSE4_INLINE M128_4Xfloat ConvertToFloat() const {
    return {_mm_cvtepi32_ps(val_)};
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    return _mm_movemask_ps(_mm_castsi128_ps(val_));
  }
};

class M128_4Xuint32 : public M128Mixin<M128_4Xuint32, uint32_t> {
 public:
  using M128Mixin = M128Mixin<M128_4Xuint32, uint32_t>;

  using M128Mixin::M128Mixin;

  SCANN_SSE4_INLINE uint32_t GetLowElement() const {
    return _mm_cvtsi128_si32(val_);
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    return _mm_movemask_ps(_mm_castsi128_ps(val_));
  }
};

class M128_8Xuint16 : public M128Mixin<M128_8Xuint16, uint16_t> {
 public:
  using M128Mixin = M128Mixin<M128_8Xuint16, uint16_t>;

  using M128Mixin::M128Mixin;
};

class M128_8Xint16 : public M128Mixin<M128_8Xint16, int16_t> {
 public:
  using M128Mixin = M128Mixin<M128_8Xint16, int16_t>;

  using M128Mixin::M128Mixin;

  SCANN_SSE4_INLINE M128_4Xint32 ExtractBotAsInt32s() const {
    return {_mm_cvtepi16_epi32(val_)};
  }

  SCANN_SSE4_INLINE M128_4Xint32 ExtractTopAsInt32s() const {
    __m128i top = _mm_srli_si128(val_, 8);
    return {_mm_cvtepi16_epi32(top)};
  }

  SCANN_SSE4_INLINE M128_8Xint16 InterleaveBottom(M128_8Xint16 x) const {
    return {_mm_unpacklo_epi16(val_, x.val_)};
  }

  SCANN_SSE4_INLINE M128_8Xint16 InterleaveTop(M128_8Xint16 x) const {
    return {_mm_unpackhi_epi16(val_, x.val_)};
  }

  SCANN_SSE4_INLINE int BitDoubledMaskFromHighBits() const {
    return _mm_movemask_epi8(val_);
  }
};

class M128_16Xuint8 : public M128Mixin<M128_16Xuint8, uint8_t> {
 public:
  using M128Mixin = M128Mixin<M128_16Xuint8, uint8_t>;

  using M128Mixin::M128Mixin;

  SCANN_SSE4_INLINE M128_8Xuint16 ExtractBotAsUint16s() const {
    return {_mm_cvtepu8_epi16(val_)};
  }

  SCANN_SSE4_INLINE M128_8Xuint16 ExtractTopAsUint16s() const {
    __m128i top = _mm_srli_si128(val_, 8);
    return {_mm_cvtepu8_epi16(top)};
  }

  SCANN_SSE4_INLINE M128_8Xint16 ExtractOddAsInt16s() const {
    return {_mm_srli_epi16(val_, 8)};
  }

  SCANN_SSE4_INLINE M128_16Xuint8
  Perform16LUT16Lookups(const M128_16Xuint8& indices) const {
    return {_mm_shuffle_epi8(val_, indices.val())};
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    return _mm_movemask_epi8(val_);
  }
};

class M128_16Xint8 : public M128Mixin<M128_16Xint8, int8_t> {
 public:
  using M128Mixin = M128Mixin<M128_16Xint8, int8_t>;

  using M128Mixin::M128Mixin;

  SCANN_SSE4_INLINE M128_8Xint16 ExtractBotAsInt16s() const {
    return {_mm_cvtepi8_epi16(val_)};
  }

  SCANN_SSE4_INLINE M128_8Xint16 ExtractTopAsInt16s() const {
    __m128i top = _mm_srli_si128(val_, 8);
    return {_mm_cvtepi8_epi16(top)};
  }

  SCANN_SSE4_INLINE M128_16Xint8
  Perform16LUT16Lookups(const M128_16Xuint8& indices) const {
    return {_mm_shuffle_epi8(val_, indices.val())};
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    return _mm_movemask_epi8(val_);
  }
};

SCANN_SSE4_INLINE M128_8Xint16 Pack32To16(M128_4Xint32 a, M128_4Xint32 b) {
  return {_mm_packs_epi32(a.val(), b.val())};
}
SCANN_SSE4_INLINE M128_8Xuint16 Pack32To16(M128_4Xuint32 a, M128_4Xuint32 b) {
  return {_mm_packus_epi32(a.val(), b.val())};
}
SCANN_SSE4_INLINE M128_16Xint8 Pack16To8(M128_8Xint16 a, M128_8Xint16 b) {
  return {_mm_packs_epi16(a.val(), b.val())};
}
SCANN_SSE4_INLINE M128_16Xuint8 Pack16To8(M128_8Xuint16 a, M128_8Xuint16 b) {
  return {_mm_packus_epi16(a.val(), b.val())};
}

SCANN_SSE4_INLINE uint32_t MaskFromHighBits(M128_8Xint16 a, M128_8Xint16 b) {
  return Pack16To8(a, b).MaskFromHighBits();
}

SCANN_SSE4_INLINE uint32_t MaskFromHighBits(M128_8Xint16 a, M128_8Xint16 b,
                                            M128_8Xint16 c, M128_8Xint16 d) {
  const uint32_t m00 = MaskFromHighBits(a, b);
  const uint32_t m16 = MaskFromHighBits(c, d);
  return m00 + (m16 << 16);
}

SCANN_SSE4_INLINE uint32_t MaskFromHighBits(M128_8Xint16 v[4]) {
  return MaskFromHighBits(v[0], v[1], v[2], v[3]);
}

SCANN_SSE4_INLINE uint32_t MaskFromHighBits(M128_4Xfloat v[8]) {
  const uint32_t m00 = v[0].MaskFromHighBits();
  const uint32_t m04 = v[1].MaskFromHighBits();
  const uint32_t m08 = v[2].MaskFromHighBits();
  const uint32_t m12 = v[3].MaskFromHighBits();
  const uint32_t m16 = v[4].MaskFromHighBits();
  const uint32_t m20 = v[5].MaskFromHighBits();
  const uint32_t m24 = v[6].MaskFromHighBits();
  const uint32_t m28 = v[7].MaskFromHighBits();
  return m00 + (m04 << 4) + (m08 << 8) + (m12 << 12) + (m16 << 16) +
         (m20 << 20) + (m24 << 24) + (m28 << 28);
}

namespace sse4 {

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
  using type = M128_4Xfloat;
};

template <typename T>
using Simd = typename SimdImpl<T>::type;

}  // namespace sse4
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
