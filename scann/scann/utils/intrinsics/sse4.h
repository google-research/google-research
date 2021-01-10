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

#ifndef SCANN_UTILS_INTRINSICS_SSE4_H_
#define SCANN_UTILS_INTRINSICS_SSE4_H_

#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <emmintrin.h>
#include <x86intrin.h>

namespace research_scann {
namespace sse4 {

static constexpr PlatformGeneration kPlatformGeneration = kBaselineSse4;

template <typename T, size_t kNumElementsRequired>
constexpr size_t InferNumRegisters() {
  constexpr size_t kRegisterBytes = 16;
  constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);

  static_assert(kNumElementsRequired > 0);
  static_assert(IsDivisibleBy(kNumElementsRequired, kElementsPerRegister));

  return kNumElementsRequired / kElementsPerRegister;
}

}  // namespace sse4

template <typename T, size_t kNumRegisters = 1, size_t... kTensorNumRegisters>
class Sse4;

struct Sse4Zeros {};
struct Sse4Uninitialized {};

template <typename T, size_t kNumRegistersInferred>
class Sse4<T, kNumRegistersInferred> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Sse4);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kRegisterBits = 128;
  static constexpr size_t kRegisterBytes = 16;
  static constexpr size_t kNumRegisters = kNumRegistersInferred;
  static constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);
  static constexpr size_t kNumElements = kNumRegisters * kElementsPerRegister;

  static auto InferIntelType() {
    if constexpr (std::is_same_v<T, float>) {
      return __m128();
    } else if constexpr (std::is_same_v<T, double>) {
      return __m128d();
    } else {
      return __m128i();
    }
  }
  using IntelType = decltype(InferIntelType());

  Sse4(Sse4Uninitialized) {}
  Sse4() : Sse4(Sse4Uninitialized()) {}

  SCANN_SSE4_INLINE Sse4(Sse4Zeros) { Clear(); }

  SCANN_SSE4_INLINE Sse4(IntelType val) {
    static_assert(kNumRegisters == 1);
    *this = val;
  }

  SCANN_SSE4_INLINE Sse4(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_SSE4_INLINE explicit Sse4(const Sse4<U, kOther>& other) {
    Sse4& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = *other[j];
      } else if constexpr (kOther == 1) {
        me[j] = *other[0];
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
  }

  SCANN_SSE4_INLINE Sse4& operator=(Sse4Zeros val) {
    Clear();
    return *this;
  }

  SCANN_SSE4_INLINE Sse4& operator=(IntelType val) {
    static_assert(kNumRegisters == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Sse4<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_SSE4_INLINE Sse4& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_SSE4_INLINE IntelType operator*() const {
    static_assert(kNumRegisters == 1);
    return registers_[0];
  }

  SCANN_SSE4_INLINE Sse4<T, 1>& operator[](size_t idx) {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  SCANN_SSE4_INLINE const Sse4<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  static SCANN_SSE4_INLINE IntelType ZeroOneRegister() {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_setzero_ps();
    } else if constexpr (IsSameAny<T, double>()) {
      return _mm_setzero_pd();
    } else {
      return _mm_setzero_si128();
    }
  }

  static SCANN_SSE4_INLINE Sse4 Zeros() {
    Sse4<T, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ZeroOneRegister();
    }
    return ret;
  }

  SCANN_SSE4_INLINE Sse4& Clear() {
    for (size_t j : Seq(kNumRegisters)) {
      registers_[j] = ZeroOneRegister();
    }
    return *this;
  }

  static SCANN_SSE4_INLINE IntelType BroadcastOneRegister(T x) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_set1_ps(x);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_set1_pd(x);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_set1_epi8(x);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_set1_epi16(x);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_set1_epi32(x);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_set1_epi64(__m64{static_cast<int64_t>(x)});
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE static Sse4 Broadcast(T x) {
    Sse4 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = BroadcastOneRegister(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  static SCANN_SSE4_INLINE IntelType LoadOneRegister(const T* address) {
    if constexpr (kAligned) {
      if constexpr (IsSameAny<T, float>()) {
        return _mm_load_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm_load_pd(address);
      } else {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(address));
      }
    } else {
      if constexpr (IsSameAny<T, float>()) {
        return _mm_loadu_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm_loadu_pd(address);
      } else {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(address));
      }
    }
  }

  template <bool kAligned = false>
  SCANN_SSE4_INLINE static Sse4 Load(const T* address) {
    Sse4 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = LoadOneRegister<kAligned>(address + j * kElementsPerRegister);
    }
    return ret;
  }

  static SCANN_SSE4_INLINE void StoreOneRegister(T* address, IntelType x) {
    if constexpr (IsSameAny<T, float>()) {
      _mm_storeu_ps(address, x);
    } else if constexpr (IsSameAny<T, double>()) {
      _mm_storeu_pd(address, x);
    } else {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(address), x);
    }
  }

  SCANN_SSE4_INLINE void Store(T* address) const {
    const Sse4& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      StoreOneRegister(address + j * kElementsPerRegister, *me[j]);
    }
  }

  SCANN_SSE4_INLINE array<T, kNumElements> Store() const {
    array<T, kNumElements> ret;
    Store(ret.data());
    return ret;
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumRegisters, kOther)>
  static SCANN_SSE4_INLINE Sse4<T, kOutput> BinaryOperatorImpl(
      const Sse4& me, const Sse4<T, kOther>& other, Op fn) {
    Sse4<T, kOutput> ret;
    for (size_t j : Seq(Sse4<T, kOutput>::kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        ret[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        ret[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        ret[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return ret;
  }

  static SCANN_SSE4_INLINE IntelType Add(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_add_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_add_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_add_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_add_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_add_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_add_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator+(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_SSE4_INLINE IntelType Subtract(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_sub_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_sub_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_sub_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_sub_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_sub_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_sub_epi64(a, b);
    }
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator-(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_SSE4_INLINE IntelType Multiply(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm_mul_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm_mul_pd(a, b);
    }

    static_assert(!IsSame<T, int8_t>(), "There's no 8-bit '*' instruction");
    if constexpr (IsSame<T, int16_t>()) {
      return _mm_mullo_epi16(a, b);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm_mullo_epi32(a, b);
    }
    static_assert(!IsSame<T, int64_t>(),
                  "_mm_mullo_epi64 is introduced in AVX-512");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Not Implemented. Unsigned multiplication is limited to "
                  "_mm_mul_epu32, which expands from uint32_t=>uint64_t.");
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator*(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_SSE4_INLINE IntelType Divide(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm_div_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm_div_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, int16_t, int32_t, int64_t>(),
                  "There's no integer '/' operations.");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "There's no integer '/' operations.");
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator/(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_SSE4_INLINE auto BitwiseAnd(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm_and_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm_and_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm_and_si128(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator&(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_SSE4_INLINE auto BitwiseOr(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm_or_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm_or_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm_or_si128(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator|(const Sse4<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_SSE4_INLINE IntelType ShiftRight(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '>>' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSame<T, int16_t>()) {
      return _mm_srai_epi16(x, count);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm_srai_epi32(x, count);
    }
    if constexpr (IsSame<T, int64_t>()) {
      return _mm_srai_epi64(x, count);
    }

    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm_srli_epi16(x, count);
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return _mm_srli_epi32(x, count);
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return _mm_srli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE Sse4 operator>>(int count) const {
    const Sse4& me = *this;
    Sse4 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_SSE4_INLINE IntelType ShiftLeft(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '<<' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_slli_epi16(x, count);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_slli_epi32(x, count);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_slli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE Sse4 operator<<(int count) const {
    const Sse4& me = *this;
    Sse4 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_SSE4_INLINE Sse4& AccumulateOperatorImpl(const Sse4<T, kOther>& other,
                                                 Op fn) {
    Sse4& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = fn(*me[j], *other[j]);
      } else if constexpr (kOther == 1) {
        me[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
    return *this;
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator+=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator-=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator*=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator/=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator&=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE Sse4& operator|=(const Sse4<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  SCANN_SSE4_INLINE Sse4& operator<<=(int count) {
    Sse4& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_SSE4_INLINE Sse4& operator>>=(int count) {
    Sse4& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumRegisters, typename Op>
  SCANN_SSE4_INLINE auto ComparisonOperatorImpl(const Sse4& me,
                                                const Sse4<T, kOther>& other,
                                                Op fn) const {
    Sse4<T, std::max(kNumRegisters, kOther)> masks;
    for (size_t j : Seq(std::max(kNumRegisters, kOther))) {
      if constexpr (kOther == kNumRegisters) {
        masks[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        masks[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        masks[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return masks;
  }

  static SCANN_SSE4_INLINE IntelType LessThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_cmplt_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_cmplt_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, int16_t, int32_t, int64_t>()) {
      return GreaterThan(b, a);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_SSE4_INLINE auto operator<(const Sse4<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_SSE4_INLINE IntelType LessOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_cmple_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_cmple_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_SSE4_INLINE auto operator<=(const Sse4<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessOrEquals);
  }

  static SCANN_SSE4_INLINE IntelType Equals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_cmpeq_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_cmpeq_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_cmpeq_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_cmpeq_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_cmpeq_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_cmpeq_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_SSE4_INLINE auto operator==(const Sse4<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_SSE4_INLINE IntelType GreaterOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_cmpge_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_cmpge_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_SSE4_INLINE auto operator>=(const Sse4<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterOrEquals);
  }

  static SCANN_SSE4_INLINE IntelType GreaterThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm_cmpgt_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm_cmpgt_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t>()) {
      return _mm_cmpgt_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t>()) {
      return _mm_cmpgt_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t>()) {
      return _mm_cmpgt_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t>()) {
      return _mm_cmpgt_epi64(a, b);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_SSE4_INLINE auto operator>(const Sse4<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  SCANN_SSE4_INLINE int MaskFromHighBits() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSame<T, float>()) {
      return _mm_movemask_ps(*me[0]);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm_movemask_pd(*me[0]);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_movemask_epi8(*me[0]);
    }
    static_assert(!IsSameAny<T, int16_t, uint16_t>(),
                  "There's no efficient single-register equivalent to the "
                  "missing _mm_movemask_epi16 op. Try the two register "
                  "MaskFromHighBits helper method.");
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_movemask_ps(_mm_castsi128_ps(*me[0]));
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm_movemask_pd(_mm_castsi128_pd(*me[0]));
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_SSE4_INLINE T GetLowElement() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSameAny<T, float, double>()) {
      return (*me[0])[0];
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_extract_epi8(*me[0], 0);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_extract_epi16(*me[0], 0);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm_cvtsi128_si32(*me[0]);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return (*me[0])[0];
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  static SCANN_SSE4_INLINE typename Sse4<U>::IntelType ConvertOneRegister(
      IntelType x) {
    if constexpr (IsSame<T, int32_t>() && IsSame<U, float>()) {
      return _mm_cvtepi32_ps(x);
    }
    static_assert(!(IsSame<T, int64_t>() && IsSame<U, double>()),
                  "_mm_cvtepi64_pd isn't defined until AVX-512");
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  SCANN_SSE4_INLINE Sse4<U, kNumRegisters> ConvertTo() const {
    const Sse4& me = *this;
    Sse4<U, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ConvertOneRegister<U>(*me[j]);
    }
    return ret;
  }

  static SCANN_SSE4_INLINE auto InferExpansionType() {
    if constexpr (IsSame<T, float>()) {
      return double();
    }
    if constexpr (IsSame<T, double>()) {
      return double();
    }

    if constexpr (IsSame<T, int8_t>()) {
      return int16_t();
    }
    if constexpr (IsSame<T, int16_t>()) {
      return int32_t();
    }
    if constexpr (IsSame<T, int32_t>()) {
      return int64_t();
    }
    if constexpr (IsSame<T, int64_t>()) {
      return int64_t();
    }

    if constexpr (IsSame<T, uint8_t>()) {
      return uint16_t();
    }
    if constexpr (IsSame<T, uint16_t>()) {
      return uint32_t();
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return uint64_t();
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return uint64_t();
    }
  }
  using ExpansionType = decltype(InferExpansionType());
  using ExpansionIntelType = typename Sse4<ExpansionType>::IntelType;
  using ExpandsTo = Sse4<ExpansionType, 2 * kNumRegisters>;

  static SCANN_SSE4_INLINE pair<ExpansionIntelType, ExpansionIntelType>
  ExpandOneRegister(IntelType x) {
    if constexpr (IsSame<T, float>()) {
      __m128 hi = _mm_srli_si128(x, 8);
      __m128 lo = x;
      return std::make_pair(_mm_cvtps_pd(lo), _mm_cvtps_pd(hi));
    }
    static_assert(!IsSame<T, double>(), "Nothing to expand to");

    if constexpr (!IsSameAny<T, float, double>()) {
      __m128 hi = _mm_srli_si128(x, 8);
      __m128 lo = x;

      if constexpr (IsSame<T, int8_t>()) {
        return std::make_pair(_mm_cvtepi8_epi16(lo), _mm_cvtepi8_epi16(hi));
      }
      if constexpr (IsSame<T, int16_t>()) {
        return std::make_pair(_mm_cvtepi16_epi32(lo), _mm_cvtepi16_epi32(hi));
      }
      if constexpr (IsSame<T, int32_t>()) {
        return std::make_pair(_mm_cvtepi32_epi64(lo), _mm_cvtepi32_epi64(hi));
      }
      static_assert(!IsSame<T, int64_t>(), "Nothing to expand to");

      if constexpr (IsSame<T, uint8_t>()) {
        return std::make_pair(_mm_cvtepu8_epi16(lo), _mm_cvtepu8_epi16(hi));
      }
      if constexpr (IsSame<T, uint16_t>()) {
        return std::make_pair(_mm_cvtepu16_epi32(lo), _mm_cvtepu16_epi32(hi));
      }
      if constexpr (IsSame<T, uint32_t>()) {
        return std::make_pair(_mm_cvtepu32_epi64(lo), _mm_cvtepu32_epi64(hi));
      }
      static_assert(!IsSame<T, uint64_t>(), "Nothing to expand to");
    }
  }

  template <typename ValidateT>
  SCANN_SSE4_INLINE ExpandsTo ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Sse4& me = *this;
    Sse4<ExpansionType, 2 * kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      pair<ExpansionIntelType, ExpansionIntelType> expanded =
          ExpandOneRegister(*me[j]);
      ret[2 * j + 0] = expanded.first;
      ret[2 * j + 1] = expanded.second;
    }
    return ret;
  }

 private:
  std::conditional_t<kNumRegisters == 1, IntelType, Sse4<T, 1>>
      registers_[kNumRegisters];

  template <typename U, size_t kOther, size_t... kTensorOther>
  friend class Sse4;
};

template <typename T, size_t kTensorNumRegisters0,
          size_t... kTensorNumRegisters>
class Sse4 {
 public:
  using SimdSubArray = Sse4<T, kTensorNumRegisters...>;

  Sse4(Sse4Uninitialized) {}
  Sse4() : Sse4(Sse4Uninitialized()) {}

  SCANN_SSE4_INLINE Sse4(Sse4Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Sse4Zeros();
    }
  }

  SCANN_SSE4_INLINE SimdSubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_SSE4_INLINE const SimdSubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_SSE4_INLINE void Store(T* address) const {
    constexpr size_t kStride =
        sizeof(decltype(SimdSubArray().Store())) / sizeof(T);
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j].Store(address + j * kStride);
    }
  }

  using StoreResultType =
      array<decltype(SimdSubArray().Store()), kTensorNumRegisters0>;
  SCANN_SSE4_INLINE StoreResultType Store() const {
    StoreResultType ret;
    for (size_t j : Seq(kTensorNumRegisters0)) {
      ret[j] = tensor_[j].Store();
    }
    return ret;
  }

 private:
  SimdSubArray tensor_[kTensorNumRegisters0];
};

template <typename T, size_t... kNumRegisters>
SCANN_SSE4_INLINE Sse4<T, index_sequence_sum_v<kNumRegisters...>> Sse4Concat(
    const Sse4<T, kNumRegisters>&... inputs) {
  Sse4<T, index_sequence_sum_v<kNumRegisters...>> ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_SSE4_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumRegisters)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <typename T, typename AllButLastSeq, size_t kLast>
struct Sse4ForImpl;

template <typename T, size_t... kAllButLast, size_t kLast>
struct Sse4ForImpl<T, index_sequence<kAllButLast...>, kLast> {
  using type = Sse4<T, kAllButLast..., sse4::InferNumRegisters<T, kLast>()>;
};

template <typename T, size_t... kTensorNumElements>
using Sse4For =
    typename Sse4ForImpl<T,
                         index_sequence_all_but_last_t<kTensorNumElements...>,
                         index_sequence_last_v<kTensorNumElements...>>::type;

static_assert(IsSame<Sse4For<uint8_t, 16>, Sse4<uint8_t>>());
static_assert(IsSame<Sse4For<uint8_t, 16>, Sse4<uint8_t, 1>>());
static_assert(IsSame<Sse4For<uint8_t, 32>, Sse4<uint8_t, 2>>());
static_assert(IsSame<Sse4For<uint64_t, 32>, Sse4<uint64_t, 16>>());

SCANN_SSE4_INLINE uint32_t GetComparisonMask(Sse4<int16_t> a, Sse4<int16_t> b) {
  return _mm_movemask_epi8(_mm_packs_epi16(*a, *b));
}

SCANN_SSE4_INLINE uint32_t GetComparisonMask(Sse4<int16_t> a, Sse4<int16_t> b,
                                             Sse4<int16_t> c, Sse4<int16_t> d) {
  const uint32_t m00 = GetComparisonMask(a, b);
  const uint32_t m16 = GetComparisonMask(c, d);
  return m00 + (m16 << 16);
}
SCANN_SSE4_INLINE uint32_t GetComparisonMask(Sse4<int16_t, 4> cmp) {
  const uint32_t m00 = GetComparisonMask(cmp[0], cmp[1]);
  const uint32_t m16 = GetComparisonMask(cmp[2], cmp[3]);
  return m00 + (m16 << 16);
}

SCANN_SSE4_INLINE uint32_t GetComparisonMask(Sse4<int16_t> cmp[2]) {
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_SSE4_INLINE int GetComparisonMask(Sse4<float> cmp) {
  return _mm_movemask_ps(*cmp);
}

SCANN_SSE4_INLINE int GetComparisonMask(Sse4<float> v00, Sse4<float> v04,
                                        Sse4<float> v08, Sse4<float> v12) {
  const uint32_t m00 = _mm_movemask_ps(*v00);
  const uint32_t m04 = _mm_movemask_ps(*v04);
  const uint32_t m08 = _mm_movemask_ps(*v08);
  const uint32_t m12 = _mm_movemask_ps(*v12);
  return m00 + (m04 << 4) + (m08 << 8) + (m12 << 12);
}

SCANN_SSE4_INLINE int GetComparisonMask(Sse4<float, 4> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
}

SCANN_SSE4_INLINE uint32_t GetComparisonMask(Sse4<float, 8> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]) +
         (GetComparisonMask(cmp[4], cmp[5], cmp[6], cmp[7]) << 16);
}

namespace sse4 {

SCANN_INLINE string_view SimdName() { return "SSE4"; }
SCANN_INLINE bool RuntimeSupportsSimd() { return RuntimeSupportsSse4(); }

template <typename T, size_t... kTensorNumRegisters>
using Simd = Sse4<T, kTensorNumRegisters...>;

template <typename T, size_t kTensorNumElements0, size_t... kTensorNumElements>
using SimdFor = Sse4For<T, kTensorNumElements0, kTensorNumElements...>;

using Zeros = Sse4Zeros;
using Uninitialized = Sse4Uninitialized;

}  // namespace sse4
}  // namespace research_scann

#else

namespace research_scann {

template <typename T, size_t... kTensorNumRegisters>
struct Sse4;

}

#endif
#endif
