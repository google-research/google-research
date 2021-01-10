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

#ifndef SCANN_UTILS_INTRINSICS_AVX2_H_
#define SCANN_UTILS_INTRINSICS_AVX2_H_

#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/avx1.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <x86intrin.h>

namespace research_scann {
namespace avx2 {

static constexpr PlatformGeneration kPlatformGeneration = kHaswellAvx2;

template <typename T, size_t kNumElementsRequired>
constexpr size_t InferNumRegisters() {
  constexpr size_t kRegisterBytes = 32;
  constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);

  static_assert(kNumElementsRequired > 0);
  static_assert(IsDivisibleBy(kNumElementsRequired, kElementsPerRegister));

  return kNumElementsRequired / kElementsPerRegister;
}

}  // namespace avx2

template <typename T, size_t kNumRegisters = 1, size_t... kTensorNumRegisters>
class Avx2;

struct Avx2Zeros {};
struct Avx2Uninitialized {};

template <typename T, size_t kNumRegistersInferred>
class Avx2<T, kNumRegistersInferred> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Avx2);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kRegisterBits = 256;
  static constexpr size_t kRegisterBytes = 32;
  static constexpr size_t kNumRegisters = kNumRegistersInferred;
  static constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);
  static constexpr size_t kNumElements = kNumRegisters * kElementsPerRegister;

  static auto InferIntelType() {
    if constexpr (std::is_same_v<T, float>) {
      return __m256();
    } else if constexpr (std::is_same_v<T, double>) {
      return __m256d();
    } else {
      return __m256i();
    }
  }
  using IntelType = decltype(InferIntelType());

  Avx2(Avx2Uninitialized) {}
  Avx2() : Avx2(Avx2Uninitialized()) {}

  SCANN_AVX2_INLINE Avx2(Avx2Zeros) { Clear(); }

  SCANN_AVX2_INLINE Avx2(IntelType val) {
    static_assert(kNumRegisters == 1);
    *this = val;
  }

  SCANN_AVX2_INLINE Avx2(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_AVX2_INLINE explicit Avx2(const Avx2<U, kOther>& other) {
    Avx2& me = *this;
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

  SCANN_AVX2_INLINE Avx2(Avx1<T, kNumRegisters> avx1) {
    Avx2& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = *avx1[j];
    }
  }

  SCANN_AVX2_INLINE operator Avx1<T, kNumRegisters>() const {
    const Avx2& me = *this;
    Avx1<T, kNumRegisters> avx1;
    for (size_t j : Seq(kNumRegisters)) {
      avx1[j] = *me[j];
    }
    return avx1;
  }

  SCANN_AVX2_INLINE Avx2& operator=(Avx2Zeros val) {
    Clear();
    return *this;
  }

  SCANN_AVX2_INLINE Avx2& operator=(IntelType val) {
    static_assert(kNumRegisters == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Avx2<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_AVX2_INLINE Avx2& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_AVX2_INLINE IntelType operator*() const {
    static_assert(kNumRegisters == 1);
    return registers_[0];
  }

  SCANN_AVX2_INLINE Avx2<T, 1>& operator[](size_t idx) {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  SCANN_AVX2_INLINE const Avx2<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  static SCANN_AVX2_INLINE IntelType ZeroOneRegister() {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_setzero_ps();
    } else if constexpr (IsSameAny<T, double>()) {
      return _mm256_setzero_ps();
    } else {
      return _mm256_setzero_si256();
    }
  }

  static SCANN_AVX2_INLINE Avx2 Zeros() {
    Avx2<T, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ZeroOneRegister();
    }
    return ret;
  }

  SCANN_AVX2_INLINE Avx2& Clear() {
    for (size_t j : Seq(kNumRegisters)) {
      registers_[j] = ZeroOneRegister();
    }
    return *this;
  }

  static SCANN_AVX2_INLINE IntelType BroadcastOneRegister(T x) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_set1_ps(x);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_set1_pd(x);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm256_set1_epi8(x);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm256_set1_epi16(x);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_set1_epi32(x);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_set1_epi64x(x);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE static Avx2 Broadcast(T x) {
    Avx2 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = BroadcastOneRegister(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  static SCANN_AVX2_INLINE IntelType LoadOneRegister(const T* address) {
    if constexpr (kAligned) {
      if constexpr (IsSameAny<T, float>()) {
        return _mm256_load_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm256_load_pd(address);
      } else {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(address));
      }
    } else {
      if constexpr (IsSameAny<T, float>()) {
        return _mm256_loadu_ps(address);
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm256_loadu_pd(address);
      } else {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(address));
      }
    }
  }

  template <bool kAligned = false>
  SCANN_AVX2_INLINE static Avx2 Load(const T* address) {
    Avx2 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = LoadOneRegister<kAligned>(address + j * kElementsPerRegister);
    }
    return ret;
  }

  static SCANN_AVX2_INLINE void StoreOneRegister(T* address, IntelType x) {
    if constexpr (IsSameAny<T, float>()) {
      _mm256_storeu_ps(address, x);
    } else if constexpr (IsSameAny<T, double>()) {
      _mm256_storeu_pd(address, x);
    } else {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(address), x);
    }
  }

  SCANN_AVX2_INLINE void Store(T* address) const {
    const Avx2& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      StoreOneRegister(address + j * kElementsPerRegister, *me[j]);
    }
  }

  SCANN_AVX2_INLINE array<T, kNumElements> Store() const {
    array<T, kNumElements> ret;
    Store(ret.data());
    return ret;
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumRegisters, kOther)>
  static SCANN_AVX2_INLINE Avx2<T, kOutput> BinaryOperatorImpl(
      const Avx2& me, const Avx2<T, kOther>& other, Op fn) {
    Avx2<T, kOutput> ret;
    for (size_t j : Seq(Avx2<T, kOutput>::kNumRegisters)) {
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

  static SCANN_AVX2_INLINE IntelType Add(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_add_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_add_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm256_add_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm256_add_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_add_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_add_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator+(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_AVX2_INLINE IntelType Subtract(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_sub_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_sub_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm256_sub_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm256_sub_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_sub_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_sub_epi64(a, b);
    }
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator-(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_AVX2_INLINE IntelType Multiply(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_mul_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_mul_pd(a, b);
    }

    static_assert(!IsSame<T, int8_t>(), "There's no 8-bit '*' instruction");
    if constexpr (IsSame<T, int16_t>()) {
      return _mm256_mullo_epi16(a, b);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm256_mullo_epi32(a, b);
    }
    static_assert(!IsSame<T, int64_t>(),
                  "_mm256_mullo_epi64 is introduced in AVX-512");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Not Implemented. Unsigned multiplication is limited to "
                  "_mm256_mul_epu32, which expands from uint32_t=>uint64_t.");
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator*(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_AVX2_INLINE IntelType Divide(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_div_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_div_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, int16_t, int32_t, int64_t>(),
                  "There's no integer '/' operations.");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "There's no integer '/' operations.");
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator/(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_AVX2_INLINE auto BitwiseAnd(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_and_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_and_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm256_and_si256(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator&(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_AVX2_INLINE auto BitwiseOr(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm256_or_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_or_pd(a, b);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return _mm256_or_si256(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator|(const Avx2<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_AVX2_INLINE IntelType ShiftRight(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '>>' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSame<T, int16_t>()) {
      return _mm256_srai_epi16(x, count);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm256_srai_epi32(x, count);
    }
    if constexpr (IsSame<T, int64_t>()) {
      return _mm256_srai_epi64(x, count);
    }

    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm256_srli_epi16(x, count);
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return _mm256_srli_epi32(x, count);
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return _mm256_srli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE Avx2 operator>>(int count) const {
    const Avx2& me = *this;
    Avx2 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_AVX2_INLINE IntelType ShiftLeft(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '<<' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm256_slli_epi16(x, count);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_slli_epi32(x, count);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_slli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX2_INLINE Avx2 operator<<(int count) const {
    const Avx2& me = *this;
    Avx2 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_AVX2_INLINE Avx2& AccumulateOperatorImpl(const Avx2<T, kOther>& other,
                                                 Op fn) {
    Avx2& me = *this;
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
  SCANN_AVX2_INLINE Avx2& operator+=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE Avx2& operator-=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE Avx2& operator*=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE Avx2& operator/=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE Avx2& operator&=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE Avx2& operator|=(const Avx2<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  SCANN_AVX2_INLINE Avx2& operator<<=(int count) {
    Avx2& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_AVX2_INLINE Avx2& operator>>=(int count) {
    Avx2& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumRegisters, typename Op>
  SCANN_AVX2_INLINE auto ComparisonOperatorImpl(const Avx2& me,
                                                const Avx2<T, kOther>& other,
                                                Op fn) const {
    Avx2<T, std::max(kNumRegisters, kOther)> masks;
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

  static SCANN_AVX2_INLINE IntelType LessThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_LT_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_LT_OS);
    }

    if constexpr (IsSameAny<T, int8_t, int16_t, int32_t, int64_t>()) {
      return GreaterThan(b, a);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX2_INLINE auto operator<(const Avx2<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_AVX2_INLINE IntelType LessOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_LE_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_LE_OS);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX2_INLINE auto operator<=(const Avx2<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessOrEquals);
  }

  static SCANN_AVX2_INLINE IntelType Equals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_EQ_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_EQ_OS);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm256_cmpeq_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm256_cmpeq_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_cmpeq_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_cmpeq_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX2_INLINE auto operator==(const Avx2<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_AVX2_INLINE IntelType GreaterOrEquals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_GE_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_GE_OS);
    }

    static_assert(!IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                             uint32_t, int64_t, uint64_t>(),
                  "Prior to AVX-512, the only integer comparison ops are '<', "
                  "'>', and '==' for signed integers.");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX2_INLINE auto operator>=(const Avx2<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterOrEquals);
  }

  static SCANN_AVX2_INLINE IntelType GreaterThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm256_cmp_ps(a, b, _CMP_GT_OS);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm256_cmp_pd(a, b, _CMP_GT_OS);
    }

    if constexpr (IsSameAny<T, int8_t>()) {
      return _mm256_cmpgt_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t>()) {
      return _mm256_cmpgt_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t>()) {
      return _mm256_cmpgt_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t>()) {
      return _mm256_cmpgt_epi64(a, b);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Prior to AVX-512, there are no unsigned comparison ops");
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX2_INLINE auto operator>(const Avx2<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  SCANN_AVX2_INLINE int MaskFromHighBits() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSame<T, float>()) {
      return _mm256_movemask_ps(*me[0]);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm256_movemask_pd(*me[0]);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm256_movemask_epi8(*me[0]);
    }
    static_assert(!IsSameAny<T, int16_t, uint16_t>(),
                  "There's no efficient single-register equivalent to the "
                  "missing _mm_movemask_epi16 op. Try the two register "
                  "MaskFromHighBits helper method.");
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_movemask_ps(_mm256_castsi256_ps(*me[0]));
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm256_movemask_pd(_mm256_castsi256_pd(*me[0]));
    }
  }

  SCANN_AVX2_INLINE T GetLowElement() const {
    static_assert(kNumRegisters == 1);
    const auto& me = *this;

    if constexpr (IsSameAny<T, float, double>()) {
      return (*me[0])[0];
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm_extract_epi8(_mm256_castsi256_si128(*me[0]), 0);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm_extract_epi16(_mm256_castsi256_si128(*me[0]), 0);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm256_cvtsi256_si32(*me[0]);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return (*me[0])[0];
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  static SCANN_AVX2_INLINE typename Avx2<U>::IntelType ConvertOneRegister(
      IntelType x) {
    if constexpr (IsSame<T, int32_t>() && IsSame<U, float>()) {
      return _mm256_cvtepi32_ps(x);
    }
    if constexpr (IsSame<T, int64_t>() && IsSame<U, double>()) {
      return _mm256_cvtepi64_pd(x);
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  SCANN_AVX2_INLINE Avx2<U, kNumRegisters> ConvertTo() const {
    const Avx2& me = *this;
    Avx2<U, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ConvertOneRegister<U>(*me[j]);
    }
    return ret;
  }

  static SCANN_AVX2_INLINE auto InferExpansionType() {
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
  using ExpansionIntelType = typename Avx2<ExpansionType>::IntelType;
  using ExpandsTo = Avx2<ExpansionType, 2 * kNumRegisters>;

  static SCANN_AVX2_INLINE pair<ExpansionIntelType, ExpansionIntelType>
  ExpandOneRegister(IntelType x) {
    if constexpr (IsSame<T, float>()) {
      __m128 hi = _mm256_extractf128_ps(x, 1);
      __m128 lo = _mm256_castps256_ps128(x);
      return std::make_pair(_mm256_cvtps_pd(lo), _mm256_cvtps_pd(hi));
    }
    static_assert(!IsSame<T, double>(), "Nothing to expand to");

    __m128i hi = _mm256_extractf128_si256(x, 1);
    __m128i lo = _mm256_castsi256_si128(x);

    if constexpr (IsSame<T, int8_t>()) {
      return std::make_pair(_mm256_cvtepi8_epi16(lo), _mm256_cvtepi8_epi16(hi));
    }
    if constexpr (IsSame<T, int16_t>()) {
      return std::make_pair(_mm256_cvtepi16_epi32(lo),
                            _mm256_cvtepi16_epi32(hi));
    }
    if constexpr (IsSame<T, int32_t>()) {
      return std::make_pair(_mm256_cvtepi32_epi64(lo),
                            _mm256_cvtepi32_epi64(hi));
    }
    static_assert(!IsSame<T, int64_t>(), "Nothing to expand to");

    if constexpr (IsSame<T, uint8_t>()) {
      return std::make_pair(_mm256_cvtepu8_epi16(lo), _mm256_cvtepu8_epi16(hi));
    }
    if constexpr (IsSame<T, uint16_t>()) {
      return std::make_pair(_mm256_cvtepu16_epi32(lo),
                            _mm256_cvtepu16_epi32(hi));
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return std::make_pair(_mm256_cvtepu32_epi64(lo),
                            _mm256_cvtepu32_epi64(hi));
    }
    static_assert(!IsSame<T, uint64_t>(), "Nothing to expand to");
  }

  template <typename ValidateT>
  SCANN_AVX2_INLINE ExpandsTo ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Avx2& me = *this;
    Avx2<ExpansionType, 2 * kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      pair<ExpansionIntelType, ExpansionIntelType> expanded =
          ExpandOneRegister(*me[j]);
      ret[2 * j + 0] = expanded.first;
      ret[2 * j + 1] = expanded.second;
    }
    return ret;
  }

 private:
  std::conditional_t<kNumRegisters == 1, IntelType, Avx2<T, 1>>
      registers_[kNumRegisters];

  template <typename U, size_t kOther, size_t... kTensorOther>
  friend class Avx2;
};

template <typename T, size_t kTensorNumRegisters0,
          size_t... kTensorNumRegisters>
class Avx2 {
 public:
  using SimdSubArray = Avx2<T, kTensorNumRegisters...>;

  Avx2(Avx2Uninitialized) {}
  Avx2() : Avx2(Avx2Uninitialized()) {}

  SCANN_AVX2_INLINE Avx2(Avx2Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Avx2Zeros();
    }
  }

  SCANN_AVX2_INLINE SimdSubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_AVX2_INLINE const SimdSubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_AVX2_INLINE void Store(T* address) const {
    constexpr size_t kStride =
        sizeof(decltype(SimdSubArray().Store())) / sizeof(T);
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j].Store(address + j * kStride);
    }
  }

  using StoreResultType =
      array<decltype(SimdSubArray().Store()), kTensorNumRegisters0>;
  SCANN_AVX2_INLINE StoreResultType Store() const {
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
SCANN_AVX2_INLINE Avx2<T, index_sequence_sum_v<kNumRegisters...>> Avx2Concat(
    const Avx2<T, kNumRegisters>&... inputs) {
  Avx2<T, index_sequence_sum_v<kNumRegisters...>> ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_AVX2_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumRegisters)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <typename T, typename AllButLastSeq, size_t kLast>
struct Avx2ForImpl;

template <typename T, size_t... kAllButLast, size_t kLast>
struct Avx2ForImpl<T, index_sequence<kAllButLast...>, kLast> {
  using type = Avx2<T, kAllButLast..., avx2::InferNumRegisters<T, kLast>()>;
};

template <typename T, size_t... kTensorNumElements>
using Avx2For =
    typename Avx2ForImpl<T,
                         index_sequence_all_but_last_t<kTensorNumElements...>,
                         index_sequence_last_v<kTensorNumElements...>>::type;

static_assert(IsSame<Avx2For<uint8_t, 32>, Avx2<uint8_t>>());
static_assert(IsSame<Avx2For<uint8_t, 32>, Avx2<uint8_t, 1>>());
static_assert(IsSame<Avx2For<uint8_t, 64>, Avx2<uint8_t, 2>>());
static_assert(IsSame<Avx2For<uint64_t, 64>, Avx2<uint64_t, 16>>());

SCANN_AVX2_INLINE uint32_t GetComparisonMask(Avx2<int16_t> a, Avx2<int16_t> b) {
  constexpr uint8_t kDestLoEqALo = 0x00;
  constexpr uint8_t kDestLoEqAHi = 0x01;
  constexpr uint8_t kDestHiEqBLo = 0x20;
  constexpr uint8_t kDestHiEqBHi = 0x30;
  constexpr uint8_t lo_spec = (kDestLoEqALo + kDestHiEqBLo);
  constexpr uint8_t hi_spec = (kDestLoEqAHi + kDestHiEqBHi);
  Avx2<int16_t> alo_blo = _mm256_permute2x128_si256(*a, *b, lo_spec);
  Avx2<int16_t> ahi_bhi = _mm256_permute2x128_si256(*a, *b, hi_spec);
  Avx2<int8_t> aa_bb = _mm256_packs_epi16(*alo_blo, *ahi_bhi);
  return _mm256_movemask_epi8(*aa_bb);
}

SCANN_AVX2_INLINE uint32_t GetComparisonMask(Avx2For<int16_t, 32> cmp) {
  static_assert(Avx2For<int16_t, 32>::kNumRegisters == 2);
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_AVX2_INLINE uint32_t GetComparisonMask(Avx2<int16_t> cmp[2]) {
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_AVX2_INLINE uint32_t GetComparisonMask(Avx2<float> v0, Avx2<float> v1,
                                             Avx2<float> v2, Avx2<float> v3) {
  const uint32_t m00 = _mm256_movemask_ps(*v0);
  const uint32_t m08 = _mm256_movemask_ps(*v1);
  const uint32_t m16 = _mm256_movemask_ps(*v2);
  const uint32_t m24 = _mm256_movemask_ps(*v3);
  return m00 + (m08 << 8) + (m16 << 16) + (m24 << 24);
}

SCANN_AVX2_INLINE uint32_t GetComparisonMask(Avx2<float, 4> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
}

namespace avx2 {

SCANN_INLINE string_view SimdName() { return "AVX2"; }
SCANN_INLINE bool RuntimeSupportsSimd() { return RuntimeSupportsAvx2(); }

template <typename T, size_t... kTensorNumRegisters>
using Simd = Avx2<T, kTensorNumRegisters...>;

template <typename T, size_t kTensorNumElements0, size_t... kTensorNumElements>
using SimdFor = Avx2For<T, kTensorNumElements0, kTensorNumElements...>;

using Zeros = Avx2Zeros;
using Uninitialized = Avx2Uninitialized;

}  // namespace avx2
}  // namespace research_scann

#else

namespace research_scann {

template <typename T, size_t... kTensorNumRegisters>
struct Avx2;

}

#endif
#endif
