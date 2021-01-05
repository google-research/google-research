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

#ifndef SCANN__UTILS_INTRINSICS_AVX512_H_
#define SCANN__UTILS_INTRINSICS_AVX512_H_

#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/avx2.h"
#include "scann/utils/types.h"

#ifdef __x86_64__

#include <x86intrin.h>

namespace tensorflow {
namespace scann_ops {
namespace avx512 {

template <typename T, size_t kNumElementsRequired>
constexpr size_t InferNumRegisters() {
  constexpr size_t kRegisterBytes = 64;
  constexpr size_t kSlotsPerRegister = kRegisterBytes / sizeof(T);

  static_assert(kNumElementsRequired > 0);
  static_assert(IsDivisibleBy(kNumElementsRequired, kSlotsPerRegister));

  return kNumElementsRequired / kSlotsPerRegister;
}

}  // namespace avx512

template <typename T = void, size_t kNumRegisters = 1,
          size_t... kTensorNumRegisters>
class Avx512;

struct Avx512Zeros {};
struct Avx512Uninitialized {};

template <typename T, size_t kNumRegistersInferred>
class Avx512<T, kNumRegistersInferred> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Avx512);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kNumRegisters = kNumRegistersInferred;
  static constexpr size_t kRegisterBits = 512;
  static constexpr size_t kRegisterBytes = 64;
  static constexpr size_t kSlotsPerRegister = kRegisterBytes / sizeof(T);
  static constexpr size_t kNumElements = kNumRegisters * kSlotsPerRegister;

  static auto InferIntelType() {
    if constexpr (std::is_same_v<T, float>) {
      return __m512();
    } else if constexpr (std::is_same_v<T, double>) {
      return __m512d();
    } else {
      return __m512i();
    }
  }
  using IntelType = decltype(InferIntelType());

  Avx512(Avx512Uninitialized) {}
  Avx512() : Avx512(Avx512Uninitialized()) {}

  SCANN_AVX512_INLINE Avx512(Avx512Zeros) { Clear(); }

  SCANN_AVX512_INLINE Avx512(IntelType val) {
    static_assert(kNumRegisters == 1);
    *this = val;
  }

  SCANN_AVX512_INLINE Avx512(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_AVX512_INLINE explicit Avx512(const Avx512<U, kOther>& other) {
    Avx512& me = *this;
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

  SCANN_AVX512_INLINE Avx512& operator=(Avx512Zeros val) {
    Clear();
    return *this;
  }

  SCANN_AVX512_INLINE Avx512& operator=(IntelType val) {
    static_assert(kNumRegisters == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Avx512<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_AVX512_INLINE Avx512& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_AVX512_INLINE IntelType operator*() const {
    static_assert(kNumRegisters == 1);
    return registers_[0];
  }

  SCANN_AVX512_INLINE Avx512<T, 1>& operator[](size_t idx) {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  SCANN_AVX512_INLINE const Avx512<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  static SCANN_AVX512_INLINE IntelType ZeroOneRegister() {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_setzero_ps();
    } else if constexpr (IsSameAny<T, double>()) {
      return _mm512_setzero_ps();
    } else {
      return _mm512_setzero_si512();
    }
  }

  static SCANN_AVX512_INLINE Avx512 Zeros() {
    Avx512<T, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ZeroOneRegister();
    }
    return ret;
  }

  SCANN_AVX512_INLINE Avx512& Clear() {
    for (size_t j : Seq(kNumRegisters)) {
      registers_[j] = ZeroOneRegister();
    }
    return *this;
  }

  static SCANN_AVX512_INLINE IntelType BroadcastOneRegister(T x) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_set1_ps(x);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm512_set1_pd(x);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm512_set1_epi8(x);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm512_set1_epi16(x);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm512_set1_epi32(x);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm512_set1_epi64(x);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX512_INLINE static Avx512 Broadcast(T x) {
    Avx512 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = BroadcastOneRegister(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  static SCANN_AVX512_INLINE IntelType LoadOneRegister(const T* address) {
    if constexpr (kAligned) {
      if constexpr (IsSameAny<T, float>()) {
        return _mm512_load_ps(reinterpret_cast<const __m512*>(address));
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm512_load_pd(reinterpret_cast<const __m512d*>(address));
      } else {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(address));
      }
    } else {
      if constexpr (IsSameAny<T, float>()) {
        return _mm512_loadu_ps(reinterpret_cast<const __m512*>(address));
      } else if constexpr (IsSameAny<T, double>()) {
        return _mm512_loadu_pd(reinterpret_cast<const __m512d*>(address));
      } else {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(address));
      }
    }
  }

  template <bool kAligned = false>
  SCANN_AVX512_INLINE static Avx512 Load(const T* address) {
    Avx512 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = LoadOneRegister<kAligned>(address + j * kSlotsPerRegister);
    }
    return ret;
  }

  static SCANN_AVX512_INLINE void StoreOneRegister(T* address, IntelType x) {
    if constexpr (IsSameAny<T, float>()) {
      _mm512_storeu_ps(reinterpret_cast<__m512*>(address), x);
    } else if constexpr (IsSameAny<T, double>()) {
      _mm512_storeu_pd(reinterpret_cast<__m512d*>(address), x);
    } else {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(address), x);
    }
  }

  SCANN_AVX512_INLINE void Store(T* address) const {
    const Avx512& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      StoreOneRegister(address + j * kSlotsPerRegister, *me[j]);
    }
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumRegisters, kOther)>
  static SCANN_AVX512_INLINE Avx512<T, kOutput> BinaryOperatorImpl(
      const Avx512& me, const Avx512<T, kOther>& other, Op fn) {
    Avx512<T, kOutput> ret;
    for (size_t j : Seq(Avx512<T, kOutput>::kNumRegisters)) {
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

  static SCANN_AVX512_INLINE IntelType Add(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_add_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm512_add_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm512_add_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm512_add_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm512_add_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm512_add_epi64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator+(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_AVX512_INLINE IntelType Subtract(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_sub_ps(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm512_sub_pd(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm512_sub_epi8(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm512_sub_epi16(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm512_sub_epi32(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm512_sub_epi64(a, b);
    }
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator-(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_AVX512_INLINE IntelType Multiply(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm512_mul_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm512_mul_pd(a, b);
    }

    static_assert(!IsSame<T, int8_t>(), "There's no 8-bit '*' instruction");
    if constexpr (IsSame<T, int16_t>()) {
      return _mm512_mullo_epi16(a, b);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm512_mullo_epi32(a, b);
    }
    if constexpr (IsSame<T, int64_t>()) {
      return _mm512_mullo_epi64(a, b);
    }

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "Not Implemented. Unsigned multiplication is limited to "
                  "_mm512_mul_epu32, which expands from uint32_t=>uint64_t.");
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator*(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_AVX512_INLINE IntelType Divide(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return _mm512_div_ps(a, b);
    }
    if constexpr (IsSame<T, double>()) {
      return _mm512_div_pd(a, b);
    }

    static_assert(!IsSameAny<T, int8_t, int16_t, int32_t, int64_t>(),
                  "There's no integer '/' operations.");

    static_assert(!IsSameAny<T, uint8_t, uint16_t, uint32_t, uint64_t>(),
                  "There's no integer '/' operations.");
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator/(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_AVX512_INLINE auto BitwiseAnd(IntelType a, IntelType b) {
    return a & b;
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator&(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_AVX512_INLINE auto BitwiseOr(IntelType a, IntelType b) {
    return a | b;
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator|(const Avx512<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_AVX512_INLINE IntelType ShiftRight(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '>>' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSame<T, int16_t>()) {
      return _mm512_srai_epi16(x, count);
    }
    if constexpr (IsSame<T, int32_t>()) {
      return _mm512_srai_epi32(x, count);
    }
    if constexpr (IsSame<T, int64_t>()) {
      return _mm512_srai_epi64(x, count);
    }

    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm512_srli_epi16(x, count);
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return _mm512_srli_epi32(x, count);
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return _mm512_srli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX512_INLINE Avx512 operator>>(int count) const {
    const Avx512& me = *this;
    Avx512 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_AVX512_INLINE IntelType ShiftLeft(IntelType x, int count) {
    static_assert(!IsSameAny<T, int8_t, uint8_t>(),
                  "There's no 8-bit '<<' instruction");
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");

    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm512_slli_epi16(x, count);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm512_slli_epi32(x, count);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm512_slli_epi64(x, count);
    }
    LOG(FATAL) << "Undefined";
  }

  SCANN_AVX512_INLINE Avx512 operator<<(int count) const {
    const Avx512& me = *this;
    Avx512 ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_AVX512_INLINE Avx512& AccumulateOperatorImpl(
      const Avx512<T, kOther>& other, Op fn) {
    Avx512& me = *this;
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
  SCANN_AVX512_INLINE Avx512& operator+=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE Avx512& operator-=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE Avx512& operator*=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE Avx512& operator/=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE Avx512& operator&=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE Avx512& operator|=(const Avx512<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  SCANN_AVX512_INLINE Avx512& operator<<=(int count) {
    Avx512& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_AVX512_INLINE Avx512& operator>>=(int count) {
    Avx512& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumRegisters, typename Op>
  SCANN_AVX512_INLINE auto ComparisonOperatorImpl(
      const Avx512& me, const Avx512<T, kOther>& other, Op fn) const {
    using MaskT = decltype(fn(*me[0], *me[0]));
    array<MaskT, std::max(kNumRegisters, kOther)> masks;
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

  static SCANN_AVX512_INLINE auto LessThan(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_cmplt_ps_mask(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm512_cmplt_pd_mask(a, b);
    }

    if constexpr (IsSameAny<T, int8_t>()) {
      return _mm512_cmplt_epi8_mask(a, b);
    }
    if constexpr (IsSameAny<T, int16_t>()) {
      return _mm512_cmplt_epi16_mask(a, b);
    }
    if constexpr (IsSameAny<T, int32_t>()) {
      return _mm512_cmplt_epi32_mask(a, b);
    }
    if constexpr (IsSameAny<T, int64_t>()) {
      return _mm512_cmplt_epi64_mask(a, b);
    }

    if constexpr (IsSameAny<T, uint8_t>()) {
      return _mm512_cmplt_epu8_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm512_cmplt_epu16_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint32_t>()) {
      return _mm512_cmplt_epu32_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint64_t>()) {
      return _mm512_cmplt_epu64_mask(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX512_INLINE auto operator<(const Avx512<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_AVX512_INLINE auto Equals(IntelType a, IntelType b) {
    if constexpr (IsSameAny<T, float>()) {
      return _mm512_cmpeq_ps_mask(a, b);
    }
    if constexpr (IsSameAny<T, double>()) {
      return _mm512_cmpeq_pd_mask(a, b);
    }

    if constexpr (IsSameAny<T, int8_t, uint8_t>()) {
      return _mm512_cmpeq_epi8_mask(a, b);
    }
    if constexpr (IsSameAny<T, int16_t, uint16_t>()) {
      return _mm512_cmpeq_epi16_mask(a, b);
    }
    if constexpr (IsSameAny<T, int32_t, uint32_t>()) {
      return _mm512_cmpeq_epi32_mask(a, b);
    }
    if constexpr (IsSameAny<T, int64_t, uint64_t>()) {
      return _mm512_cmpeq_epi64_mask(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther>
  SCANN_AVX512_INLINE auto operator==(const Avx512<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_AVX512_INLINE auto GreaterThan(IntelType a, IntelType b) {
    static_assert(!IsSameAny<T, float, double>(),
                  "There's no floating point '>' instruction.");

    if constexpr (IsSameAny<T, int8_t>()) {
      return _mm512_cmpgt_epi8_mask(a, b);
    }
    if constexpr (IsSameAny<T, int16_t>()) {
      return _mm512_cmpgt_epi16_mask(a, b);
    }
    if constexpr (IsSameAny<T, int32_t>()) {
      return _mm512_cmpgt_epi32_mask(a, b);
    }
    if constexpr (IsSameAny<T, int64_t>()) {
      return _mm512_cmpgt_epi64(a, b);
    }

    if constexpr (IsSameAny<T, uint8_t>()) {
      return _mm512_cmpgt_epu8_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint16_t>()) {
      return _mm512_cmpgt_epu16_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint32_t>()) {
      return _mm512_cmpgt_epu32_mask(a, b);
    }
    if constexpr (IsSameAny<T, uint64_t>()) {
      return _mm512_cmpgt_epu64(a, b);
    }
    LOG(FATAL) << "Undefined";
  }

  template <size_t kOther = kNumRegisters>
  SCANN_AVX512_INLINE auto operator>(const Avx512<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  template <typename U>
  static SCANN_AVX512_INLINE typename Avx512<U>::IntelType ConvertOneRegister(
      IntelType x) {
    if constexpr (IsSame<T, int32_t>() && IsSame<U, float>()) {
      return _mm512_cvtepi32_ps(x);
    }
    if constexpr (IsSame<T, int64_t>() && IsSame<U, double>()) {
      return _mm512_cvtepi64_pd(x);
    }
    LOG(FATAL) << "Undefined";
  }

  template <typename U>
  SCANN_AVX512_INLINE Avx512<U, kNumRegisters> ConvertTo() const {
    const Avx512& me = *this;
    Avx512<U, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ConvertOneRegister<U>(*me[j]);
    }
    return ret;
  }

  static SCANN_AVX512_INLINE auto InferExpansionType() {
    if constexpr (IsSame<T, float>()) {
      return double();
    }
    if constexpr (IsSame<T, double>()) {
      return void();
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
      return void();
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
      return void();
    }
  }
  using ExpansionType = decltype(InferExpansionType());
  using ExpansionIntelType = typename Avx512<ExpansionType>::IntelType;
  using ExpandsTo = Avx512<ExpansionType, 2 * kNumRegisters>;

  static SCANN_AVX512_INLINE pair<ExpansionIntelType, ExpansionIntelType>
  ExpandOneRegister(IntelType x) {
    if constexpr (IsSame<T, float>()) {
      __m256 hi = _mm512_extractf32x8_ps(x, 1);
      __m256 lo = _mm512_castps512_ps256(x);
      return std::make_pair(_mm512_cvtps_pd(lo), _mm512_cvtps_pd(hi));
    }
    static_assert(!IsSame<T, double>(), "Nothing to expand to");

    __m256i lo = _mm512_castsi512_si256(x);
    __m256i hi = _mm512_extracti64x4_epi64(x, 1);

    if constexpr (IsSame<T, int8_t>()) {
      return std::make_pair(_mm512_cvtepi8_epi16(lo), _mm512_cvtepi8_epi16(hi));
    }
    if constexpr (IsSame<T, int16_t>()) {
      return std::make_pair(_mm512_cvtepi16_epi32(lo),
                            _mm512_cvtepi16_epi32(hi));
    }
    if constexpr (IsSame<T, int32_t>()) {
      return std::make_pair(_mm512_cvtepi32_epi64(lo),
                            _mm512_cvtepi32_epi64(hi));
    }
    static_assert(!IsSame<T, int64_t>(), "Nothing to expand to");

    if constexpr (IsSame<T, uint8_t>()) {
      return std::make_pair(_mm512_cvtepu8_epi16(lo), _mm512_cvtepu8_epi16(hi));
    }
    if constexpr (IsSame<T, uint16_t>()) {
      return std::make_pair(_mm512_cvtepu16_epi32(lo),
                            _mm512_cvtepu16_epi32(hi));
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return std::make_pair(_mm512_cvtepu32_epi64(lo),
                            _mm512_cvtepu32_epi64(hi));
    }
    static_assert(!IsSame<T, uint64_t>(), "Nothing to expand to");
  }

  template <typename ValidateT>
  SCANN_AVX512_INLINE ExpandsTo ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Avx512& me = *this;
    Avx512<ExpansionType, 2 * kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      pair<ExpansionIntelType, ExpansionIntelType> expanded =
          ExpandOneRegister(*me[j]);
      ret[2 * j + 0] = expanded.first;
      ret[2 * j + 1] = expanded.second;
    }
    return ret;
  }

 private:
  std::conditional_t<kNumRegisters == 1, IntelType, Avx512<T, 1>>
      registers_[kNumRegisters];

  template <typename U, size_t kOther, size_t... kTensorOther>
  friend class Avx512;
};

template <size_t kNumRegisters>
class Avx512<void, kNumRegisters> {
 public:
  static constexpr size_t kRegisterBits = 512;
  static constexpr size_t kRegisterBytes = 64;

  using IntelType = __m512i;

  static constexpr Avx512Zeros Zeros() { return {}; }

  static constexpr Avx512Uninitialized Uninitialized() { return {}; }

 private:
  Avx512() {}
};

template <typename T, size_t kTensorNumRegisters0,
          size_t... kTensorNumRegisters>
class Avx512 {
 public:
  using Avx512SubArray = Avx512<T, kTensorNumRegisters...>;

  Avx512(Avx512Uninitialized) {}
  Avx512() : Avx512(Avx512Uninitialized()) {}

  SCANN_AVX512_INLINE Avx512(Avx512Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Avx512Zeros();
    }
  }

  SCANN_AVX512_INLINE Avx512SubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_AVX512_INLINE const Avx512SubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

 private:
  Avx512SubArray tensor_[kTensorNumRegisters0];
};

template <size_t... kInts>
constexpr size_t index_sequence_sum_v = (kInts + ...);

template <typename T, size_t... kNumRegisters>
SCANN_AVX512_INLINE Avx512<T, index_sequence_sum_v<kNumRegisters...>>
Avx512Concat(const Avx512<T, kNumRegisters>&... inputs) {
  Avx512<T, index_sequence_sum_v<kNumRegisters...>> ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_AVX512_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumRegisters)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <size_t... kInts>
constexpr size_t index_sequence_last_v = (kInts, ...);

template <typename Seq0, typename Seq1>
struct index_sequence_minus_last_impl;

template <size_t... kFirst, size_t kMiddle, size_t... kLast>
struct index_sequence_minus_last_impl<std::index_sequence<kFirst...>,
                                      std::index_sequence<kMiddle, kLast...>> {
  using type = typename index_sequence_minus_last_impl<
      std::index_sequence<kFirst..., kMiddle>,
      std::index_sequence<kLast...>>::type;
};

template <size_t... kFirst, size_t kLast>
struct index_sequence_minus_last_impl<std::index_sequence<kFirst...>,
                                      std::index_sequence<kLast>> {
  using type = std::index_sequence<kFirst...>;
};

template <size_t... kInts>
using index_sequence_minus_last_t = typename index_sequence_minus_last_impl<
    std::index_sequence<>, std::index_sequence<kInts...>>::type;

static_assert(
    IsSame<index_sequence_minus_last_t<1, 2, 3>, std::index_sequence<1, 2>>());

static_assert(IsSame<index_sequence_minus_last_t<3, 5, 4, 7>,
                     std::index_sequence<3, 5, 4>>());

template <typename T = void, size_t kNumRegisters0 = 1,
          size_t... kTensorNumRegisters>
using Avx512Array = Avx512<T, kNumRegisters0, kTensorNumRegisters...>;

template <typename T, typename TensorSeq, size_t kTensorNumRegisters9>
struct Avx512ForImpl;

template <typename T, size_t... kTensorNumRegisters,
          size_t kTensorNumRegisters9>
struct Avx512ForImpl<T, std::index_sequence<kTensorNumRegisters...>,
                     kTensorNumRegisters9> {
  using type = Avx512<T, kTensorNumRegisters...,
                      avx512::InferNumRegisters<T, kTensorNumRegisters9>()>;
};

template <typename T, size_t... kTensorNumRegisters>
using Avx512For =
    typename Avx512ForImpl<T,
                           index_sequence_minus_last_t<kTensorNumRegisters...>,
                           index_sequence_last_v<kTensorNumRegisters...>>::type;

static_assert(IsSame<Avx512For<uint8_t, 64>, Avx512<uint8_t>>());
static_assert(IsSame<Avx512For<uint8_t, 64>, Avx512<uint8_t, 1>>());
static_assert(IsSame<Avx512For<uint8_t, 128>, Avx512<uint8_t, 2>>());
static_assert(IsSame<Avx512For<uint64_t, 128>, Avx512<uint64_t, 16>>());

SCANN_AVX512_INLINE __m512i fake_mm512_permutex2var_epi128(
    __m512i a, __m512i b, array<uint64_t, 4> lane_idxs) {
  Avx512<uint64_t, 1> idxs = _mm512_set_epi64(
      2 * lane_idxs[3] + 1, 2 * lane_idxs[3] + 0, 2 * lane_idxs[2] + 1,
      2 * lane_idxs[2] + 0, 2 * lane_idxs[1] + 1, 2 * lane_idxs[1] + 0,
      2 * lane_idxs[0] + 1, 2 * lane_idxs[0] + 0);
  return _mm512_permutex2var_epi64(a, *idxs, b);
}

SCANN_AVX512_INLINE __m512i fake_mm512_permutex2var_epi256(
    __m512i a, __m512i b, array<uint64_t, 2> lane_idxs) {
  return fake_mm512_permutex2var_epi128(
      a, b,
      {2 * lane_idxs[0] + 0, 2 * lane_idxs[0] + 1, 2 * lane_idxs[1] + 0,
       2 * lane_idxs[1] + 1});
}

namespace avx512 {

template <typename T>
using Simd = Avx512<T>;

using Zeros = Avx512Zeros;
using Uninitialized = Avx512Uninitialized;

}  // namespace avx512
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
