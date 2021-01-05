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

#ifndef SCANN__UTILS_INTRINSICS_AVX2_H_
#define SCANN__UTILS_INTRINSICS_AVX2_H_

#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/avx1.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <x86intrin.h>

namespace tensorflow {
namespace scann_ops {

template <typename ClassName, typename CType>
class M256Mixin {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(M256Mixin);

  SCANN_INLINE static constexpr size_t BlockSize() {
    return 32 / sizeof(CType);
  }

  SCANN_AVX2_INLINE M256Mixin() {}

  SCANN_AVX2_INLINE M256Mixin(__m256i val) : val_(val) {}

  SCANN_AVX2_INLINE static ClassName Zeros() {
    return {_mm256_setzero_si256()};
  }

  SCANN_AVX2_INLINE static ClassName Broadcast(CType x) {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_set1_epi8(x)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_set1_epi16(x)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_set1_epi32(x)};
    }
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit broadcast instruction");
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE __m256i val() const { return val_; }

  SCANN_AVX2_INLINE static ClassName Load(const CType* address) {
    return ClassName(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(address)));
  }

  SCANN_AVX2_INLINE void Store(CType* address) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(address), val_);
  }

  SCANN_AVX2_INLINE ClassName operator+(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_add_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_add_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_add_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_add_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator-(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_sub_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_sub_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_sub_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_sub_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator>>(int count) const {
    static_assert(!IsSameAny<CType, int8_t, uint8_t>(),
                  "There is no 8-bit '>>' instruction");
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit '>>' instruction");

    if (IsSame<CType, int16_t>()) {
      return {_mm256_srai_epi16(val_, count)};
    }
    if (IsSame<CType, int32_t>()) {
      return {_mm256_srai_epi32(val_, count)};
    }

    if (IsSameAny<CType, uint16_t>()) {
      return {_mm256_srli_epi16(val_, count)};
    }
    if (IsSame<CType, uint32_t>()) {
      return {_mm256_srli_epi32(val_, count)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator<<(int count) const {
    static_assert(!IsSameAny<CType, int8_t, uint8_t>(),
                  "There is no 8-bit '<<' instruction");
    static_assert(!IsSameAny<CType, int64_t, uint64_t>(),
                  "There is no 64-bit '<<' instruction");

    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_slli_epi16(val_, count)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_slli_epi32(val_, count)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator==(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_cmpeq_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_cmpeq_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_cmpeq_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_cmpeq_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator>(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_cmpgt_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_cmpgt_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_cmpgt_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_cmpgt_epi64(val_, other.val_)};
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE ClassName operator<(const ClassName& other) const {
    return (other > *reinterpret_cast<const ClassName*>(this));
  }

  SCANN_AVX2_INLINE ClassName operator&(const ClassName& other) const {
    return {val_ & other.val_};
  }

  SCANN_AVX2_INLINE ClassName operator|(const ClassName& other) const {
    return {val_ | other.val_};
  }

  SCANN_AVX2_INLINE ClassName& operator+=(const ClassName& other) {
    val_ = (*this + other).val();
    return *reinterpret_cast<ClassName*>(this);
  }

  SCANN_AVX2_INLINE ClassName& operator-=(const ClassName& other) {
    val_ = (*this - other).val();
    return *reinterpret_cast<ClassName*>(this);
  }

  SCANN_AVX2_INLINE ClassName& operator>>=(int count) {
    val_ = (*this >> count).val();
    return *reinterpret_cast<ClassName*>(this);
  }

  SCANN_AVX2_INLINE ClassName
  InterleaveBottomPerLane(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_unpacklo_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_unpacklo_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_unpacklo_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_unpacklo_epi64(val_, other.val_)};
    }
  }

  SCANN_AVX2_INLINE ClassName
  InterleaveTopPerLane(const ClassName& other) const {
    if (IsSameAny<CType, int8_t, uint8_t>()) {
      return {_mm256_unpackhi_epi8(val_, other.val_)};
    }
    if (IsSameAny<CType, int16_t, uint16_t>()) {
      return {_mm256_unpackhi_epi16(val_, other.val_)};
    }
    if (IsSameAny<CType, int32_t, uint32_t>()) {
      return {_mm256_unpackhi_epi32(val_, other.val_)};
    }
    if (IsSameAny<CType, int64_t, uint64_t>()) {
      return {_mm256_unpackhi_epi64(val_, other.val_)};
    }
  }

 private:
  __m256i val_;
  friend ClassName;
};

class M256_8Xint32 : public M256Mixin<M256_8Xint32, int32_t> {
 public:
  SCANN_AVX2_INLINE M256_8Xint32() {}

  SCANN_AVX2_INLINE M256_8Xint32(__m256i val) : M256Mixin(val) {}

  SCANN_AVX2_INLINE M256_8Xfloat ConvertToFloat() const {
    return {_mm256_cvtepi32_ps(val_)};
  }
};

class M256_8Xuint32 : public M256Mixin<M256_8Xuint32, uint32_t> {
 public:
  SCANN_AVX2_INLINE M256_8Xuint32() {}

  SCANN_AVX2_INLINE M256_8Xuint32(__m256i val) : M256Mixin(val) {}

  SCANN_AVX2_INLINE uint32_t GetLowElement() const {
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(val_));
  }

  SCANN_AVX2_INLINE M256_8Xfloat ReinterpretCastAsFloat() const {
    return {_mm256_castsi256_ps(val_)};
  }

  SCANN_AVX2_INLINE int MaskFromHighBits() const {
    return ReinterpretCastAsFloat().MaskFromHighBits();
  }
};

class M256_16Xuint16 : public M256Mixin<M256_16Xuint16, uint16_t> {
 public:
  SCANN_AVX2_INLINE M256_16Xuint16() {}

  SCANN_AVX2_INLINE M256_16Xuint16(__m256i val) : M256Mixin(val) {}
};

class M256_16Xint16 : public M256Mixin<M256_16Xint16, int16_t> {
 public:
  SCANN_AVX2_INLINE M256_16Xint16() {}

  SCANN_AVX2_INLINE M256_16Xint16(__m256i val) : M256Mixin(val) {}

  SCANN_AVX2_INLINE M256_8Xint32 ExtractBotAs8Xint32() const {
    return {_mm256_cvtepi16_epi32(_mm256_castsi256_si128(val_))};
  }

  SCANN_AVX2_INLINE M256_8Xint32 ExtractTopAs8Xint32() const {
    return {_mm256_cvtepi16_epi32(_mm256_extractf128_si256(val_, 1))};
  }

  SCANN_AVX2_INLINE int BitDoubledMaskFromHighBits() const {
    return _mm256_movemask_epi8(val_);
  }
};

class M256_32Xuint8 : public M256Mixin<M256_32Xuint8, uint8_t> {
 public:
  SCANN_AVX2_INLINE M256_32Xuint8() {}

  SCANN_AVX2_INLINE M256_32Xuint8(__m256i val) : M256Mixin(val) {}

  SCANN_AVX2_INLINE M256_16Xint16 ExtractBotPerLaneAs16Xint16() const {
    return {_mm256_unpacklo_epi8(val_, _mm256_setzero_si256())};
  }

  SCANN_AVX2_INLINE M256_16Xint16 ExtractTopPerLaneAs16Xint16() const {
    return {_mm256_unpackhi_epi8(val_, _mm256_setzero_si256())};
  }

  SCANN_AVX2_INLINE M256_16Xint16 ExtractOddAs16Xint16() const {
    return {_mm256_srli_epi16(val_, 8)};
  }

  SCANN_AVX2_INLINE M256_32Xuint8
  Perform32LUT16Lookups(const M256_32Xuint8& indices) const {
    return {_mm256_shuffle_epi8(val_, indices.val())};
  }

  SCANN_AVX2_INLINE int MaskFromHighBits() const {
    return _mm256_movemask_epi8(val_);
  }
};

class M256_32Xint8 : public M256Mixin<M256_32Xint8, int8_t> {
 public:
  SCANN_AVX2_INLINE M256_32Xint8() {}

  SCANN_AVX2_INLINE M256_32Xint8(__m256i val) : M256Mixin(val) {}

  SCANN_AVX2_INLINE M256_32Xint8
  Perform32LUT16Lookups(const M256_32Xuint8& indices) const {
    return {_mm256_shuffle_epi8(val_, indices.val())};
  }

  SCANN_AVX2_INLINE int MaskFromHighBits() const {
    return _mm256_movemask_epi8(val_);
  }
};

SCANN_AVX2_INLINE M256_16Xint16 Pack32To16(M256_8Xint32 a, M256_8Xint32 b) {
  return {_mm256_packs_epi32(a.val(), b.val())};
}
SCANN_AVX2_INLINE M256_16Xuint16 Pack32To16(M256_8Xuint32 a, M256_8Xuint32 b) {
  return {_mm256_packus_epi32(a.val(), b.val())};
}
SCANN_AVX2_INLINE M256_32Xint8 Pack16To8(M256_16Xint16 a, M256_16Xint16 b) {
  return {_mm256_packs_epi16(a.val(), b.val())};
}
SCANN_AVX2_INLINE M256_32Xuint8 Pack16To8(M256_16Xuint16 a, M256_16Xuint16 b) {
  return {_mm256_packus_epi16(a.val(), b.val())};
}

SCANN_AVX2_INLINE uint32_t MaskFromHighBits(M256_16Xint16 a, M256_16Xint16 b) {
  constexpr uint8_t kDestLoEqALo = 0x00;
  constexpr uint8_t kDestLoEqAHi = 0x01;
  constexpr uint8_t kDestHiEqBLo = 0x20;
  constexpr uint8_t kDestHiEqBHi = 0x30;
  constexpr uint8_t lo_spec = (kDestLoEqALo + kDestHiEqBLo);
  constexpr uint8_t hi_spec = (kDestLoEqAHi + kDestHiEqBHi);
  M256_16Xint16 alo_blo = _mm256_permute2x128_si256(a.val(), b.val(), lo_spec);
  M256_16Xint16 ahi_bhi = _mm256_permute2x128_si256(a.val(), b.val(), hi_spec);
  M256_32Xint8 aa_bb = Pack16To8(alo_blo, ahi_bhi);
  return aa_bb.MaskFromHighBits();
}

SCANN_AVX2_INLINE uint32_t MaskFromHighBits(M256_16Xint16 v[2]) {
  return MaskFromHighBits(v[0], v[1]);
}

namespace avx2 {

template <typename T>
struct SimdImpl;

template <>
struct SimdImpl<int8_t> {
  using type = M256_32Xint8;
};

template <>
struct SimdImpl<uint8_t> {
  using type = M256_32Xuint8;
};

template <>
struct SimdImpl<int16_t> {
  using type = M256_16Xint16;
};

template <>
struct SimdImpl<uint16_t> {
  using type = M256_16Xuint16;
};

template <>
struct SimdImpl<int32_t> {
  using type = M256_8Xint32;
};

template <>
struct SimdImpl<uint32_t> {
  using type = M256_8Xuint32;
};

template <>
struct SimdImpl<float> {
  using type = M256_8Xfloat;
};

template <typename T>
using Simd = typename SimdImpl<T>::type;

}  // namespace avx2
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
