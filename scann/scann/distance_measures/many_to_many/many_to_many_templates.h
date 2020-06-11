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



#ifndef SCANN__DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_
#define SCANN__DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

#ifdef __x86_64__
#include <immintrin.h>
#endif

#define SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_5(batch_size, function, ...) \
  switch (batch_size) {                                                   \
    case 0:                                                               \
      break;                                                              \
    case 1:                                                               \
      function<1>(__VA_ARGS__);                                           \
      break;                                                              \
    case 2:                                                               \
      function<2>(__VA_ARGS__);                                           \
      break;                                                              \
    case 3:                                                               \
      function<3>(__VA_ARGS__);                                           \
      break;                                                              \
    case 4:                                                               \
      function<4>(__VA_ARGS__);                                           \
      break;                                                              \
    case 5:                                                               \
      function<5>(__VA_ARGS__);                                           \
      break;                                                              \
    default:                                                              \
      LOG(FATAL) << "Invalid Batch Size";                                 \
  }
#define SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6(batch_size, function, ...) \
  switch (batch_size) {                                                   \
    case 0:                                                               \
      break;                                                              \
    case 1:                                                               \
      function<1>(__VA_ARGS__);                                           \
      break;                                                              \
    case 2:                                                               \
      function<2>(__VA_ARGS__);                                           \
      break;                                                              \
    case 3:                                                               \
      function<3>(__VA_ARGS__);                                           \
      break;                                                              \
    case 4:                                                               \
      function<4>(__VA_ARGS__);                                           \
      break;                                                              \
    case 5:                                                               \
      function<5>(__VA_ARGS__);                                           \
      break;                                                              \
    case 6:                                                               \
      function<6>(__VA_ARGS__);                                           \
      break;                                                              \
    default:                                                              \
      LOG(FATAL) << "Invalid Batch Size";                                 \
  }

namespace tensorflow {
namespace scann_ops {
namespace mm_internal {

#ifdef __x86_64__

enum : int {
  kDestLoEqALo = 0x00,
  kDestLoEqAHi = 0x01,
  kDestHiEqBLo = 0x20,
  kDestHiEqBHi = 0x30,
  t1spec = (kDestLoEqALo + kDestHiEqBHi),
  t2spec = (kDestLoEqAHi + kDestHiEqBLo)
};

SCANN_AVX1_INLINE __m256 Sum128BitLanes(__m256 a, __m256 b) {
  auto term0 = _mm256_permute2f128_ps(a, b, t1spec);
  auto term1 = _mm256_permute2f128_ps(a, b, t2spec);
  return term0 + term1;
}

SCANN_AVX1_INLINE __m256 Sum64BitLanes(__m256 a, __m256 b) {
  constexpr int kShuffleMask1 = 0b11100100;
  constexpr int kShuffleMask2 = 0b01001110;
  return _mm256_shuffle_ps(a, b, kShuffleMask1) +
         _mm256_shuffle_ps(a, b, kShuffleMask2);
}

SCANN_AVX1_INLINE void Sum8_4(__m256 a, __m256 b, __m256 c, __m256 d,
                              float* resulta, float* resultb, float* resultc,
                              float* resultd) {
  auto ac = Sum128BitLanes(a, c);
  auto bd = Sum128BitLanes(b, d);
  auto abcd = Sum64BitLanes(ac, bd);

  constexpr int kShuffleMask = 0b11110101;
  abcd += _mm256_shuffle_ps(abcd, abcd, kShuffleMask);
  *resulta = abcd[0];
  *resultb = abcd[2];
  *resultc = abcd[4];
  *resultd = abcd[6];
}

SCANN_AVX1_INLINE void Sum8_2(__m256 a, __m256 b, float* resulta,
                              float* resultb) {
  auto sum = Sum128BitLanes(a, b);

  constexpr int kShuffleMask1 = 0b11101110;
  sum += _mm256_shuffle_ps(sum, sum, kShuffleMask1);

  constexpr int kShuffleMask2 = 0b11100101;
  sum += _mm256_shuffle_ps(sum, sum, kShuffleMask2);
  *resulta = sum[0];
  *resultb = sum[4];
}

SCANN_AVX1_INLINE void Sum4_2(__m256d a, __m256d b, double* resulta,
                              double* resultb) {
  auto term0 = _mm256_permute2f128_pd(a, b, t1spec);
  auto term1 = _mm256_permute2f128_pd(a, b, t2spec);
  auto sum = term0 + term1;

  constexpr int kShuffleMask1 = 0b1111;
  sum += _mm256_shuffle_pd(sum, sum, kShuffleMask1);

  *resulta = sum[0];
  *resultb = sum[2];
}

struct AvxFloatBase {
 public:
  using Accumulator = __m256;
  static constexpr size_t kBlockSize = 8;
  SCANN_AVX1_INLINE static __m256 Loadu(const float* f) {
    return _mm256_loadu_ps(f);
  }
  SCANN_AVX1_INLINE static void Storeu(float* p, __m256 acc) {
    return _mm256_storeu_ps(p, acc);
  }
  SCANN_AVX1_INLINE static __m256 Broadcast(const float* f) {
    return _mm256_broadcast_ss(f);
  }
  SCANN_AVX1_INLINE static __m256 Multiply(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, b);
  }
  SCANN_AVX1_INLINE static __m256 Add(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
  }
  SCANN_AVX1_INLINE static __m256 Zeros() { return _mm256_setzero_ps(); }
  SCANN_AVX1_INLINE static void Postprocess2SimdRegisters(__m256 a, __m256 b,
                                                          float* resulta,
                                                          float* resultb) {
    return Sum8_2(a, b, resulta, resultb);
  }
  SCANN_AVX1_INLINE static void Postprocess4SimdRegisters(
      __m256 a, __m256 b, __m256 c, __m256 d, float* resulta, float* resultb,
      float* resultc, float* resultd) {
    return Sum8_4(a, b, c, d, resulta, resultb, resultc, resultd);
  }
};

struct AvxDoubleBase {
  using Accumulator = __m256d;
  static constexpr size_t kBlockSize = 4;
  SCANN_AVX1_INLINE static __m256d Loadu(const double* f) {
    return _mm256_loadu_pd(f);
  }
  SCANN_AVX1_INLINE static void Storeu(double* p, __m256d acc) {
    return _mm256_storeu_pd(p, acc);
  }
  SCANN_AVX1_INLINE static __m256d Broadcast(const double* f) {
    return _mm256_broadcast_sd(f);
  }
  SCANN_AVX1_INLINE static __m256d Multiply(__m256d a, __m256d b) {
    return _mm256_mul_pd(a, b);
  }
  SCANN_AVX1_INLINE static __m256d Add(__m256d a, __m256d b) {
    return _mm256_add_pd(a, b);
  }
  SCANN_AVX1_INLINE static __m256d Zeros() { return _mm256_setzero_pd(); }
  SCANN_AVX1_INLINE static void Postprocess2SimdRegisters(__m256d a, __m256d b,
                                                          double* resulta,
                                                          double* resultb) {
    Sum4_2(a, b, resulta, resultb);
  }
  SCANN_AVX1_INLINE static void Postprocess4SimdRegisters(
      __m256d a, __m256d b, __m256d c, __m256d d, double* resulta,
      double* resultb, double* resultc, double* resultd) {
    Postprocess2SimdRegisters(a, b, resulta, resultb);
    Postprocess2SimdRegisters(c, d, resultc, resultd);
  }
};

namespace avx512 {

struct Avx512FloatBase {
 public:
  using Accumulator = __m512;
  static constexpr size_t kBlockSize = 16;
  SCANN_AVX512_INLINE static __m512 Loadu(const float* f) {
    return _mm512_loadu_ps(f);
  }
  SCANN_AVX512_INLINE static void Storeu(float* p, __m512 acc) {
    return _mm512_storeu_ps(p, acc);
  }
  SCANN_AVX512_INLINE static __m512 Broadcast(const float* f) {
    return _mm512_broadcast_f32x8(AvxFloatBase::Broadcast(f));
  }
  SCANN_AVX512_INLINE static __m512 Multiply(__m512 a, __m512 b) {
    return _mm512_mul_ps(a, b);
  }
  SCANN_AVX512_INLINE static __m512 Add(__m512 a, __m512 b) {
    return _mm512_add_ps(a, b);
  }
  SCANN_AVX512_INLINE static __m512 Zeros() { return _mm512_setzero_ps(); }
  SCANN_AVX512_INLINE static void Postprocess2SimdRegisters(__m512 a, __m512 b,
                                                            float* resulta,
                                                            float* resultb) {
    *resulta = _mm512_reduce_add_ps(a);
    *resultb = _mm512_reduce_add_ps(b);
  }
  SCANN_AVX512_INLINE static void Postprocess4SimdRegisters(
      __m512 a, __m512 b, __m512 c, __m512 d, float* resulta, float* resultb,
      float* resultc, float* resultd) {
    *resulta = _mm512_reduce_add_ps(a);
    *resultb = _mm512_reduce_add_ps(b);
    *resultc = _mm512_reduce_add_ps(c);
    *resultd = _mm512_reduce_add_ps(d);
  }
};

struct Avx512DoubleBase {
  using Accumulator = __m512d;
  static constexpr size_t kBlockSize = 8;
  SCANN_AVX512_INLINE static __m512d Loadu(const double* f) {
    return _mm512_loadu_pd(f);
  }
  SCANN_AVX512_INLINE static void Storeu(double* p, __m512d acc) {
    return _mm512_storeu_pd(p, acc);
  }
  SCANN_AVX512_INLINE static __m512d Broadcast(const double* f) {
    return _mm512_broadcast_f64x4(AvxDoubleBase::Broadcast(f));
  }
  SCANN_AVX512_INLINE static __m512d Multiply(__m512d a, __m512d b) {
    return _mm512_mul_pd(a, b);
  }
  SCANN_AVX512_INLINE static __m512d Add(__m512d a, __m512d b) {
    return _mm512_add_pd(a, b);
  }
  SCANN_AVX512_INLINE static __m512d Zeros() { return _mm512_setzero_pd(); }
  SCANN_AVX512_INLINE static void Postprocess2SimdRegisters(__m512d a,
                                                            __m512d b,
                                                            double* resulta,
                                                            double* resultb) {
    *resulta = _mm512_reduce_add_pd(a);
    *resultb = _mm512_reduce_add_pd(b);
  }
  SCANN_AVX512_INLINE static void Postprocess4SimdRegisters(
      __m512d a, __m512d b, __m512d c, __m512d d, double* resulta,
      double* resultb, double* resultc, double* resultd) {
    *resulta = _mm512_reduce_add_pd(a);
    *resultb = _mm512_reduce_add_pd(b);
    *resultc = _mm512_reduce_add_pd(c);
    *resultd = _mm512_reduce_add_pd(d);
  }
};

template <typename T>
struct DotProductDistanceFunctions;

template <>
struct DotProductDistanceFunctions<float> : public Avx512FloatBase {
  SCANN_AVX512_INLINE static void Accumulate(__m512 a, __m512 b, __m512* acc) {
    *acc = _mm512_fnmadd_ps(a, b, *acc);
  }
  SCANN_AVX512_INLINE static void Accumulate(float a, float b, float* acc) {
    *acc -= a * b;
  }
};

template <>
struct DotProductDistanceFunctions<double> : public Avx512DoubleBase {
  SCANN_AVX512_INLINE static void Accumulate(__m512d a, __m512d b,
                                             __m512d* acc) {
    *acc = _mm512_fnmadd_pd(a, b, *acc);
  }
  SCANN_AVX512_INLINE static void Accumulate(double a, double b, double* acc) {
    *acc -= a * b;
  }
};

template <typename T>
struct SquaredL2DistanceFunctions;

template <>
struct SquaredL2DistanceFunctions<float> : public Avx512FloatBase {
  SCANN_AVX512_INLINE static void Accumulate(__m512 a, __m512 b, __m512* acc) {
    __m512 diff = _mm512_sub_ps(a, b);
    *acc = _mm512_fmadd_ps(diff, diff, *acc);
  }
  SCANN_AVX512_INLINE static void Accumulate(float a, float b, float* acc) {
    const float diff = a - b;
    *acc += diff * diff;
  }
};

template <>
struct SquaredL2DistanceFunctions<double> : public Avx512DoubleBase {
  SCANN_AVX512_INLINE static void Accumulate(__m512d a, __m512d b,
                                             __m512d* acc) {
    __m512d diff = _mm512_sub_pd(a, b);
    *acc = _mm512_fmadd_pd(diff, diff, *acc);
  }
  SCANN_AVX512_INLINE static void Accumulate(double a, double b, double* acc) {
    const double diff = a - b;
    *acc += diff * diff;
  }
};

}  // namespace avx512

namespace avx2 {

template <typename T>
struct DotProductDistanceFunctions;

template <>
struct DotProductDistanceFunctions<float> : public AvxFloatBase {
  SCANN_AVX2_INLINE static void Accumulate(__m256 a, __m256 b, __m256* acc) {
    *acc = _mm256_fnmadd_ps(a, b, *acc);
  }
  SCANN_AVX2_INLINE static void Accumulate(float a, float b, float* acc) {
    *acc -= a * b;
  }
};

template <>
struct DotProductDistanceFunctions<double> : public AvxDoubleBase {
  SCANN_AVX2_INLINE static void Accumulate(__m256d a, __m256d b, __m256d* acc) {
    *acc = _mm256_fnmadd_pd(a, b, *acc);
  }
  SCANN_AVX2_INLINE static void Accumulate(double a, double b, double* acc) {
    *acc -= a * b;
  }
};

template <typename T>
struct SquaredL2DistanceFunctions;

template <>
struct SquaredL2DistanceFunctions<float> : public AvxFloatBase {
  SCANN_AVX2_INLINE static void Accumulate(__m256 a, __m256 b, __m256* acc) {
    __m256 diff = _mm256_sub_ps(a, b);
    *acc = _mm256_fmadd_ps(diff, diff, *acc);
  }
  SCANN_AVX2_INLINE static void Accumulate(float a, float b, float* acc) {
    const float diff = a - b;
    *acc += diff * diff;
  }
};

template <>
struct SquaredL2DistanceFunctions<double> : public AvxDoubleBase {
  SCANN_AVX2_INLINE static void Accumulate(__m256d a, __m256d b, __m256d* acc) {
    __m256d diff = _mm256_sub_pd(a, b);
    *acc = _mm256_fmadd_pd(diff, diff, *acc);
  }
  SCANN_AVX2_INLINE static void Accumulate(double a, double b, double* acc) {
    const double diff = a - b;
    *acc += diff * diff;
  }
};

}  // namespace avx2

namespace avx1 {

template <typename T>
struct DotProductDistanceFunctions;

template <>
struct DotProductDistanceFunctions<float> : public AvxFloatBase {
  SCANN_AVX1_INLINE static void Accumulate(__m256 a, __m256 b, __m256* acc) {
    *acc = _mm256_sub_ps(*acc, _mm256_mul_ps(a, b));
  }
  SCANN_AVX1_INLINE static void Accumulate(float a, float b, float* acc) {
    *acc -= a * b;
  }
};

template <>
struct DotProductDistanceFunctions<double> : public AvxDoubleBase {
  SCANN_AVX1_INLINE static void Accumulate(__m256d a, __m256d b, __m256d* acc) {
    *acc = _mm256_sub_pd(*acc, _mm256_mul_pd(a, b));
  }
  SCANN_AVX1_INLINE static void Accumulate(double a, double b, double* acc) {
    *acc -= a * b;
  }
};

template <typename T>
struct SquaredL2DistanceFunctions;

template <>
struct SquaredL2DistanceFunctions<float> : public AvxFloatBase {
  SCANN_AVX1_INLINE static void Accumulate(__m256 a, __m256 b, __m256* acc) {
    __m256 diff = _mm256_sub_ps(a, b);
    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(diff, diff));
  }
  SCANN_AVX1_INLINE static void Accumulate(float a, float b, float* acc) {
    const float diff = a - b;
    *acc += diff * diff;
  }
};

template <>
struct SquaredL2DistanceFunctions<double> : public AvxDoubleBase {
  SCANN_AVX1_INLINE static void Accumulate(__m256d a, __m256d b, __m256d* acc) {
    __m256d diff = _mm256_sub_pd(a, b);
    *acc = _mm256_add_pd(*acc, _mm256_mul_pd(diff, diff));
  }
  SCANN_AVX1_INLINE static void Accumulate(double a, double b, double* acc) {
    const double diff = a - b;
    *acc += diff * diff;
  }
};

}  // namespace avx1

namespace sse4 {

SCANN_INLINE float HorizontalSumFloat(__m128 x) {
  constexpr int kShuffleMask1 = 0b11101110;
  x += _mm_shuffle_ps(x, x, kShuffleMask1);

  constexpr int kShuffleMask2 = 0b11100101;
  x += _mm_shuffle_ps(x, x, kShuffleMask2);
  return x[0];
}

SCANN_INLINE double HorizontalSumDouble(__m128d x) {
  constexpr int kShuffleMask1 = 0b11;
  x += _mm_shuffle_pd(x, x, kShuffleMask1);
  return x[0];
}

struct Sse4FloatBase {
  using Accumulator = __m128;
  static constexpr size_t kBlockSize = 4;
  SCANN_INLINE static __m128 Loadu(const float* f) { return _mm_loadu_ps(f); }
  SCANN_INLINE static void Storeu(float* p, __m128 acc) {
    return _mm_storeu_ps(p, acc);
  }
  SCANN_INLINE static __m128 Broadcast(const float* f) {
    return _mm_load1_ps(f);
  }
  SCANN_INLINE static __m128 Multiply(__m128 a, __m128 b) {
    return _mm_mul_ps(a, b);
  }
  SCANN_INLINE static __m128 Add(__m128 a, __m128 b) {
    return _mm_add_ps(a, b);
  }
  SCANN_INLINE static __m128 Zeros() { return _mm_setzero_ps(); }
  SCANN_INLINE static void Postprocess2SimdRegisters(__m128 a, __m128 b,
                                                     float* resulta,
                                                     float* resultb) {
    *resulta = HorizontalSumFloat(a);
    *resultb = HorizontalSumFloat(b);
  }
  SCANN_INLINE static void Postprocess4SimdRegisters(
      __m128 a, __m128 b, __m128 c, __m128 d, float* resulta, float* resultb,
      float* resultc, float* resultd) {
    Postprocess2SimdRegisters(a, b, resulta, resultb);
    Postprocess2SimdRegisters(c, d, resultc, resultd);
  }
};

struct Sse4DoubleBase {
  using Accumulator = __m128d;
  static constexpr size_t kBlockSize = 2;
  SCANN_INLINE static __m128d Loadu(const double* f) { return _mm_loadu_pd(f); }
  SCANN_INLINE static void Storeu(double* p, __m128d acc) {
    return _mm_storeu_pd(p, acc);
  }
  SCANN_INLINE static __m128d Broadcast(const double* f) {
    return _mm_load1_pd(f);
  }
  SCANN_INLINE static __m128d Multiply(__m128d a, __m128d b) {
    return _mm_mul_pd(a, b);
  }
  SCANN_INLINE static __m128d Add(__m128d a, __m128d b) {
    return _mm_add_pd(a, b);
  }
  SCANN_INLINE static __m128d Zeros() { return _mm_setzero_pd(); }
  SCANN_INLINE static void Postprocess2SimdRegisters(__m128d a, __m128d b,
                                                     double* resulta,
                                                     double* resultb) {
    *resulta = HorizontalSumDouble(a);
    *resultb = HorizontalSumDouble(b);
  }
  SCANN_INLINE static void Postprocess4SimdRegisters(
      __m128d a, __m128d b, __m128d c, __m128d d, double* resulta,
      double* resultb, double* resultc, double* resultd) {
    Postprocess2SimdRegisters(a, b, resulta, resultb);
    Postprocess2SimdRegisters(c, d, resultc, resultd);
  }
};

template <typename T>
struct DotProductDistanceFunctions;

template <>
struct DotProductDistanceFunctions<float> : public Sse4FloatBase {
  SCANN_INLINE static void Accumulate(__m128 a, __m128 b, __m128* acc) {
    *acc = _mm_sub_ps(*acc, _mm_mul_ps(a, b));
  }
  SCANN_INLINE static void Accumulate(float a, float b, float* acc) {
    *acc -= a * b;
  }
};

template <>
struct DotProductDistanceFunctions<double> : public Sse4DoubleBase {
  SCANN_INLINE static void Accumulate(__m128d a, __m128d b, __m128d* acc) {
    *acc = _mm_sub_pd(*acc, _mm_mul_pd(a, b));
  }
  SCANN_INLINE static void Accumulate(double a, double b, double* acc) {
    *acc -= a * b;
  }
};

template <typename T>
struct SquaredL2DistanceFunctions;

template <>
struct SquaredL2DistanceFunctions<float> : public Sse4FloatBase {
  SCANN_INLINE static void Accumulate(__m128 a, __m128 b, __m128* acc) {
    __m128 diff = _mm_sub_ps(a, b);
    *acc = _mm_add_ps(*acc, _mm_mul_ps(diff, diff));
  }
  SCANN_INLINE static void Accumulate(float a, float b, float* acc) {
    const float diff = a - b;
    *acc += diff * diff;
  }
};

template <>
struct SquaredL2DistanceFunctions<double> : public Sse4DoubleBase {
  SCANN_INLINE static void Accumulate(__m128d a, __m128d b, __m128d* acc) {
    __m128d diff = _mm_sub_pd(a, b);
    *acc = _mm_add_pd(*acc, _mm_mul_pd(diff, diff));
  }
  SCANN_INLINE static void Accumulate(double a, double b, double* acc) {
    const double diff = a - b;
    *acc += diff * diff;
  }
};

}  // namespace sse4

#endif

namespace portable {

template <typename FloatT>
struct PortableBase {
  using Accumulator = FloatT;
  static constexpr size_t kBlockSize = 1;
  static FloatT Loadu(const FloatT* f) { return *f; }
  static void Storeu(FloatT* p, FloatT f) { *p = f; }
  static FloatT Broadcast(const FloatT* f) { return *f; }
  static FloatT Multiply(FloatT a, FloatT b) { return a * b; }
  static FloatT Add(FloatT a, FloatT b) { return a + b; }
  static FloatT Zeros() { return 0; }
  static void Postprocess2SimdRegisters(FloatT a, FloatT b, FloatT* resulta,
                                        FloatT* resultb) {
    *resulta = a;
    *resultb = b;
  }
  static void Postprocess4SimdRegisters(FloatT a, FloatT b, FloatT c, FloatT d,
                                        FloatT* resulta, FloatT* resultb,
                                        FloatT* resultc, FloatT* resultd) {
    *resulta = a;
    *resultb = b;
    *resultc = c;
    *resultd = d;
  }
};

template <typename FloatT>
struct DotProductDistanceFunctions : public PortableBase<FloatT> {
  static void Accumulate(FloatT a, FloatT b, FloatT* acc) { *acc -= a * b; }
};

template <typename FloatT>
struct SquaredL2DistanceFunctions : public PortableBase<FloatT> {
  static void Accumulate(FloatT a, FloatT b, FloatT* acc) {
    const FloatT diff = (a - b);
    const FloatT squared_l2 = (diff * diff);
    *acc += squared_l2;
  }
};

}  // namespace portable

#define SCANN_SIMD_INLINE SCANN_SIMD_ATTRIBUTE SCANN_INLINE
#define SCANN_SIMD_INLINE_LAMBDA SCANN_SIMD_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_SIMD_OUTLINE SCANN_SIMD_ATTRIBUTE SCANN_OUTLINE

#ifdef __x86_64__

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512_ATTRIBUTE
#define SCANN_FP8_EXPAND_WIDTH 16
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_FP8_EXPAND_WIDTH
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2_ATTRIBUTE
#define SCANN_FP8_EXPAND_WIDTH 8
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_FP8_EXPAND_WIDTH
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace avx1 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1_ATTRIBUTE

#define SCANN_FP8_EXPAND_WIDTH 4
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_FP8_EXPAND_WIDTH
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE
#define SCANN_FP8_EXPAND_WIDTH 4
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_FP8_EXPAND_WIDTH
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4

#endif

namespace portable {
#define SCANN_SIMD_ATTRIBUTE
#define SCANN_FP8_EXPAND_WIDTH 1
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_FP8_EXPAND_WIDTH
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace portable

namespace internal {

inline bool IsSupportedDistanceMeasure(const DistanceMeasure& dist) {
  switch (dist.specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
    case DistanceMeasure::SQUARED_L2:
    case DistanceMeasure::COSINE:
      return true;
    default:
      return false;
  }
}

template <typename FloatT, typename CallbackT>
void CallOneToManyDistance(const DistanceMeasure& dist,
                           const DenseDataset<FloatT>& queries,
                           const DenseDataset<FloatT>& database,
                           thread::ThreadPool* pool, CallbackT callback) {
  auto one_query_results_storage = make_unique<FloatT[]>(database.size());
  MutableSpan<FloatT> one_query_results(one_query_results_storage.get(),
                                        database.size());
  for (size_t query_idx : IndicesOf(queries)) {
    DenseDistanceOneToMany(dist, queries[query_idx], database,
                           one_query_results, pool);
    callback(one_query_results, 0, query_idx);
  }
}

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl2(
    const DistanceMeasure& dist, const DenseDataset<FloatT>& queries,
    const DenseDataset<FloatT>& database, thread::ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");
  DCHECK_GE(queries.size(), 2);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                               callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else if (RuntimeSupportsAvx1()) {
    return avx1::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else if (RuntimeSupportsSse4()) {
    return sse4::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  }
#endif

  return portable::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                               std::move(callback));
}

template <typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyFP8PretransposedImpl2(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, thread::ThreadPool* pool,
    CallbackT callback) {
  DCHECK_GE(queries.size(), 1);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                       pool, callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else if (RuntimeSupportsAvx1()) {
    return avx1::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else if (RuntimeSupportsSse4()) {
    return sse4::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  }
#endif

  return portable::DenseManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}

}  // namespace internal

template <typename FloatT, typename CallbackT>
void DenseDistanceManyToManyImpl(const DistanceMeasure& dist,
                                 const DenseDataset<FloatT>& queries,
                                 const DenseDataset<FloatT>& database,
                                 thread::ThreadPool* pool, CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");
  if (queries.empty()) return;

  if (queries.size() == 1 || !internal::IsSupportedDistanceMeasure(dist)) {
    return internal::CallOneToManyDistance(dist, queries, database, pool,
                                           std::move(callback));
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper =
        [&callback](MutableSpan<FloatT> block_distances,
                    DatapointIndex base_dp_idx, DatapointIndex query_idx) {
          for (auto& elem : block_distances) {
            elem += static_cast<FloatT>(1.0);
          }
          callback(block_distances, base_dp_idx, query_idx);
        };
    return internal::DenseDistanceManyToManyImpl2<FloatT>(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    return internal::DenseDistanceManyToManyImpl2<FloatT, CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
}

template <typename CallbackT>
Status DenseDistanceManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, thread::ThreadPool* pool,
    CallbackT callback) {
  if (queries.empty()) return OkStatus();

  if (!internal::IsSupportedDistanceMeasure(dist)) {
    return InvalidArgumentError(
        "DenseDistanceManyToManyFP8Pretransposed only supports dot product, "
        "cosine and squared L2 distance.");
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper = [&callback](MutableSpan<float> block_distances,
                                             DatapointIndex base_dp_idx,
                                             DatapointIndex query_idx) {
      for (auto& elem : block_distances) {
        elem += static_cast<float>(1.0);
      }
      callback(block_distances, base_dp_idx, query_idx);
    };
    internal::DenseDistanceManyToManyFP8PretransposedImpl2(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    internal::DenseDistanceManyToManyFP8PretransposedImpl2<CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
  return OkStatus();
}

#undef SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_5
#undef SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6

}  // namespace mm_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
