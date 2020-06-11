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



#ifndef SCANN__UTILS_REDUCTION_H_
#define SCANN__UTILS_REDUCTION_H_

#include <sys/types.h>

#include "scann/data_format/datapoint.h"
#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {
namespace scann_ops {

template <typename T, typename U, typename Reduce>
auto DensePairAccumulate(T* a, U* b, size_t size, Reduce reduce)
    -> AccumulatorTypeFor<T, U> {
  using AT = AccumulatorTypeFor<T, U>;
  AT result0 = 0;
  AT result1 = 0;
  AT result2 = 0;
  AT result3 = 0;

  auto enda = a + size;
  for (; a + 3 < enda; a += 4, b += 4) {
    reduce(&result0, a[0], b[0]);
    reduce(&result1, a[1], b[1]);
    reduce(&result2, a[2], b[2]);
    reduce(&result3, a[3], b[3]);
  }

  result2 += result3;
  if (a + 1 < enda) {
    reduce(&result0, a[0], b[0]);
    reduce(&result1, a[1], b[1]);
    a += 2;
    b += 2;
  }

  result1 += result2;
  if (a < enda) {
    reduce(&result0, a[0], b[0]);
  }

  return result0 + result1;
}

template <typename I1, typename V1, typename I2, typename V2,
          typename ReduceTwo>
auto SparsePairAccumulateImpl1(I1 indices1, V1 values1, size_t nonzero_entries1,
                               I2 indices2, V2 values2, size_t nonzero_entries2,
                               ReduceTwo reduce_two)
    -> AccumulatorTypeFor<decltype(values1[0]), decltype(values2[0])> {
  using AT = AccumulatorTypeFor<decltype(values1[0]), decltype(values2[0])>;

  if (nonzero_entries1 == 0 || nonzero_entries2 == 0) return 0;

  AT result = 0;

  size_t i1_front = 0, i2_front = 0;
  size_t i1_back = nonzero_entries1 - 1, i2_back = nonzero_entries2 - 1;

  while (i1_front < i1_back && i2_front < i2_back) {
    const size_t to_add_front1 = indices1[i1_front] <= indices2[i2_front];
    const size_t to_add_front2 = indices1[i1_front] >= indices2[i2_front];
    const size_t to_sub_back2 = indices1[i1_back] <= indices2[i2_back];
    const size_t to_sub_back1 = indices1[i1_back] >= indices2[i2_back];
    if (ABSL_PREDICT_FALSE(indices1[i1_front] == indices2[i2_front])) {
      reduce_two(&result, values1[i1_front], values2[i2_front]);
    }

    if (ABSL_PREDICT_FALSE(indices1[i1_back] == indices2[i2_back])) {
      reduce_two(&result, values1[i1_back], values2[i2_back]);
    }

    i1_front += to_add_front1;
    i2_front += to_add_front2;
    i1_back -= to_sub_back1;
    i2_back -= to_sub_back2;
  }

  if (i1_front == i1_back) {
    for (; i2_front <= i2_back; ++i2_front) {
      if (ABSL_PREDICT_FALSE(indices1[i1_front] == indices2[i2_front])) {
        reduce_two(&result, values1[i1_front], values2[i2_front]);
        break;
      }
    }
  } else if (i2_front == i2_back) {
    for (; i1_front <= i1_back; ++i1_front) {
      if (ABSL_PREDICT_FALSE(indices1[i1_front] == indices2[i2_front])) {
        reduce_two(&result, values1[i1_front], values2[i2_front]);
        break;
      }
    }
  }

  return result;
}

template <typename T>
inline T* PunInt(T* arg) {
  return arg;
}
inline int32_t* PunInt(float* arg) { return reinterpret_cast<int32_t*>(arg); }
inline int64_t* PunInt(double* arg) { return reinterpret_cast<int64_t*>(arg); }

template <typename I1, typename V1, typename I2, typename V2,
          typename ReduceTwo, typename ReduceOne>
auto SparsePairAccumulateImpl2(I1 indices1, V1 values1, size_t nonzero_entries1,
                               I2 indices2, V2 values2,
                               const size_t nonzero_entries2,
                               ReduceTwo reduce_two, ReduceOne reduce_one)
    -> AccumulatorTypeFor<decltype(values1[0]), decltype(values2[0])> {
  using AT = AccumulatorTypeFor<decltype(values1[0]), decltype(values2[0])>;
  AT result0 = 0, result1 = 0;

  ssize_t i1_front = 0, i2_front = 0;
  ssize_t i1_back = nonzero_entries1, i2_back = nonzero_entries2;
  --i1_back;
  --i2_back;

  while (i1_front < i1_back && i2_front < i2_back) {
    auto front_left = values1[i1_front];
    auto front_right = values2[i2_front];
    auto back_left = values1[i1_back];
    auto back_right = values2[i2_back];

    const size_t to_add_front1 = indices1[i1_front] <= indices2[i2_front];
    const size_t to_add_front2 = indices1[i1_front] >= indices2[i2_front];
    const size_t to_sub_back2 = indices1[i1_back] <= indices2[i2_back];
    const size_t to_sub_back1 = indices1[i1_back] >= indices2[i2_back];

    *PunInt(&front_left) &= -to_add_front1;
    *PunInt(&front_right) &= -to_add_front2;
    *PunInt(&back_left) &= -to_sub_back1;
    *PunInt(&back_right) &= -to_sub_back2;

    reduce_two(&result0, front_left, front_right);
    reduce_two(&result1, back_left, back_right);
    i1_front += to_add_front1;
    i2_front += to_add_front2;
    i1_back -= to_sub_back1;
    i2_back -= to_sub_back2;
  }

  while (i1_front <= i1_back && i2_front <= i2_back) {
    if (indices1[i1_front] == indices2[i2_front]) {
      reduce_two(&result0, values1[i1_front++], values2[i2_front++]);
    } else if (indices1[i1_front] < indices2[i2_front]) {
      reduce_one(&result0, values1[i1_front++]);
    } else {
      reduce_one(&result0, values2[i2_front++]);
    }
  }

  if (i1_front > i1_back) {
    for (; i2_front <= i2_back; ++i2_front) {
      reduce_one(&result0, values2[i2_front]);
    }
  } else {
    for (; i1_front <= i1_back; ++i1_front) {
      reduce_one(&result0, values1[i1_front]);
    }
  }

  return result0 + result1;
}

template <typename T, typename U, typename ReduceTwo, typename ReduceOne>
inline AccumulatorTypeFor<T, U> SparsePairAccumulate(const DatapointPtr<T>& a,
                                                     const DatapointPtr<U>& b,
                                                     ReduceTwo reduce_two,
                                                     ReduceOne reduce_one) {
  if (reduce_one.IsNoop()) {
    return SparsePairAccumulateImpl1(
        a.indices(), a.values(), a.nonzero_entries(), b.indices(), b.values(),
        b.nonzero_entries(), reduce_two);
  } else {
    return SparsePairAccumulateImpl2(
        a.indices(), a.values(), a.nonzero_entries(), b.indices(), b.values(),
        b.nonzero_entries(), reduce_two, reduce_one);
  }
}

template <typename I1, typename V1, typename I2, typename V2,
          typename ReduceTwo, typename ReduceOne>
auto SparsePairAccumulate(I1 indices1, V1 values1, size_t nonzero_entries1,
                          I2 indices2, V2 values2,
                          const size_t nonzero_entries2, ReduceTwo reduce_two,
                          ReduceOne reduce_one)
    -> AccumulatorTypeFor<decltype(values1[0]), decltype(values2[0])> {
  if (reduce_one.IsNoop()) {
    return SparsePairAccumulateImpl1(indices1, values1, nonzero_entries1,
                                     indices2, values2, nonzero_entries2,
                                     reduce_two);
  } else {
    return SparsePairAccumulateImpl2(indices1, values1, nonzero_entries1,
                                     indices2, values2, nonzero_entries2,
                                     reduce_two, reduce_one);
  }
}

template <typename T, typename U, typename ReduceTwo>
inline AccumulatorTypeFor<T, U> HybridPairAccumulateImpl1(
    const DatapointPtr<T>& a, const DatapointPtr<U>& b, ReduceTwo reduce_two) {
  DCHECK(a.IsSparse());
  DCHECK(b.IsDense());
  auto sparse_index_ptr = a.indices();
  auto sparse_values_ptr = a.values();
  const auto sparse_index_end = sparse_index_ptr + a.nonzero_entries();
  auto dense_values_ptr = b.values();
  using AT = AccumulatorTypeFor<T, U>;
  AT result0 = 0;
  AT result1 = 0;
  AT result2 = 0;
  AT result3 = 0;

  for (; sparse_index_ptr + 3 < sparse_index_end;
       sparse_index_ptr += 4, sparse_values_ptr += 4) {
    reduce_two(&result0, dense_values_ptr[sparse_index_ptr[0]],
               sparse_values_ptr[0]);
    reduce_two(&result1, dense_values_ptr[sparse_index_ptr[1]],
               sparse_values_ptr[1]);
    reduce_two(&result2, dense_values_ptr[sparse_index_ptr[2]],
               sparse_values_ptr[2]);
    reduce_two(&result3, dense_values_ptr[sparse_index_ptr[3]],
               sparse_values_ptr[3]);
  }

  result2 += result3;
  if (sparse_index_ptr + 1 < sparse_index_end) {
    reduce_two(&result0, dense_values_ptr[sparse_index_ptr[0]],
               sparse_values_ptr[0]);
    reduce_two(&result1, dense_values_ptr[sparse_index_ptr[1]],
               sparse_values_ptr[1]);
    sparse_index_ptr += 2;
    sparse_values_ptr += 2;
  }

  result1 += result2;
  if (sparse_index_ptr < sparse_index_end) {
    reduce_two(&result0, dense_values_ptr[sparse_index_ptr[0]],
               sparse_values_ptr[0]);
  }

  return result0 + result1;
}

template <typename T, typename U, typename ReduceTwo, typename ReduceOne>
AccumulatorTypeFor<T, U> HybridPairAccumulateImpl2(const DatapointPtr<T>& a,
                                                   const DatapointPtr<U>& b,
                                                   ReduceTwo reduce_two,
                                                   ReduceOne reduce_one) {
  DCHECK(a.IsSparse());
  DCHECK(b.IsDense());
  using AT = AccumulatorTypeFor<T, U>;
  AT result0 = 0;
  AT result1 = 0;
  AT result2 = 0;
  AT result3 = 0;

  auto cur_dense = b.values();
  auto end_dense = cur_dense + b.nonzero_entries();
  for (; cur_dense + 3 < end_dense; cur_dense += 4) {
    reduce_one(&result0, *cur_dense);
    reduce_one(&result1, cur_dense[1]);
    reduce_one(&result2, cur_dense[2]);
    reduce_one(&result3, cur_dense[3]);
  }

  if (cur_dense + 1 < end_dense) {
    reduce_one(&result0, *cur_dense);
    reduce_one(&result1, cur_dense[1]);
    cur_dense += 2;
  }

  if (cur_dense < end_dense) {
    reduce_one(&result0, *cur_dense);
  }

  AT throw_away0 = 0;
  AT throw_away1 = 0;
  AT throw_away2 = 0;
  AT throw_away3 = 0;

  auto sparse_index_ptr = a.indices();
  auto sparse_value_ptr = a.values();
  auto sparse_index_end = a.indices() + a.nonzero_entries();
  for (; sparse_index_ptr + 3 < sparse_index_end;
       sparse_index_ptr += 4, sparse_value_ptr += 4) {
    reduce_one(&throw_away0, b.values()[sparse_index_ptr[0]]);
    reduce_two(&result0, b.values()[sparse_index_ptr[0]], *sparse_value_ptr);
    reduce_one(&throw_away1, b.values()[sparse_index_ptr[1]]);
    reduce_two(&result1, b.values()[sparse_index_ptr[1]], sparse_value_ptr[1]);
    reduce_one(&throw_away2, b.values()[sparse_index_ptr[2]]);
    reduce_two(&result2, b.values()[sparse_index_ptr[2]], sparse_value_ptr[2]);
    reduce_one(&throw_away3, b.values()[sparse_index_ptr[3]]);
    reduce_two(&result3, b.values()[sparse_index_ptr[3]], sparse_value_ptr[3]);
  }

  if (sparse_index_ptr + 1 < sparse_index_end) {
    reduce_one(&throw_away0, b.values()[sparse_index_ptr[0]]);
    reduce_two(&result0, b.values()[sparse_index_ptr[0]], *sparse_value_ptr);
    reduce_one(&throw_away1, b.values()[sparse_index_ptr[1]]);
    reduce_two(&result1, b.values()[sparse_index_ptr[1]], sparse_value_ptr[1]);
    sparse_index_ptr += 2;
    sparse_value_ptr += 2;
  }

  if (sparse_index_ptr < sparse_index_end) {
    reduce_one(&throw_away0, b.values()[sparse_index_ptr[0]]);
    reduce_two(&result0, b.values()[sparse_index_ptr[0]], *sparse_value_ptr);
  }

  return ((result0 - throw_away0) + (result1 - throw_away1)) +
         ((result2 - throw_away2) + (result3 - throw_away3));
}

template <typename T, typename U, typename ReduceTwo, typename ReduceOne>
AccumulatorTypeFor<T, U> HybridPairAccumulate(const DatapointPtr<T>& a,
                                              const DatapointPtr<U>& b,
                                              ReduceTwo reduce_two,
                                              ReduceOne reduce_one) {
  DCHECK(a.IsSparse() != b.IsSparse());
  if (reduce_one.IsNoop()) {
    if (a.IsSparse()) {
      return HybridPairAccumulateImpl1(a, b, reduce_two);
    } else {
      return HybridPairAccumulateImpl1(b, a, reduce_two);
    }
  } else {
    if (a.IsSparse()) {
      return HybridPairAccumulateImpl2(a, b, reduce_two, reduce_one);
    } else {
      return HybridPairAccumulateImpl2(b, a, reduce_two, reduce_one);
    }
  }
}

template <typename T, typename Reduce>
AccumulatorTypeFor<T> DenseSingleAccumulate(ConstSpan<T> values,
                                            Reduce reduce) {
  using AT = AccumulatorTypeFor<T>;
  AT result0 = 0;
  AT result1 = 0;
  AT result2 = 0;
  AT result3 = 0;

  const T* ptr = values.data();
  const T* const end = values.data() + values.size();
  for (; ptr + 4 <= end; ptr += 4) {
    reduce(&result0, ptr[0]);
    reduce(&result1, ptr[1]);
    reduce(&result2, ptr[2]);
    reduce(&result3, ptr[3]);
  }

  result2 += result3;
  if (ptr + 2 <= end) {
    reduce(&result0, ptr[0]);
    reduce(&result1, ptr[1]);
    ptr += 2;
  }

  result1 += result2;
  if (ptr < end) {
    reduce(&result0, ptr[0]);
  }

  return result0 + result1;
}

struct DoNothingReduce {
  template <typename... T>
  void operator()(T... args) {}
  bool IsNoop() { return true; }
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
