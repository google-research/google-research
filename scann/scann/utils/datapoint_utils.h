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



#ifndef SCANN__UTILS_DATAPOINT_UTILS_H_
#define SCANN__UTILS_DATAPOINT_UTILS_H_

#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/proto/input_output.pb.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
inline T RoundIfFixedPoint(float num) {
  return static_cast<T>(std::round(num));
}
template <typename T>
inline T RoundIfFixedPoint(double num) {
  return static_cast<T>(std::round(num));
}
template <>
inline float RoundIfFixedPoint<float>(float num) {
  return num;
}
template <>
inline double RoundIfFixedPoint<double>(float num) {
  return num;
}
template <>
inline float RoundIfFixedPoint<float>(double num) {
  return num;
}
template <>
inline double RoundIfFixedPoint<double>(double num) {
  return num;
}

template <typename T, typename U>
bool HaveIntersectingNonzeroes(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b);

template <typename T>
inline void NormalizeUnitL2(Datapoint<T>* dp);

template <typename To, typename From>
inline void NormalizeUnitL2(const DatapointPtr<From>& input,
                            Datapoint<To>* output);

template <typename T>
inline Status NormalizeByTag(Normalization tag, Datapoint<T>* dp);

template <typename T>
double Sum(const DatapointPtr<T>& dptr);

template <typename T>
inline void MeanVar(const DatapointPtr<T>& input, double* mean, double* var);

template <typename To, typename From>
inline void CopyToDatapoint(const DatapointPtr<From>& input,
                            Datapoint<To>* output);

template <typename To, typename From>
inline DatapointPtr<To> MaybeConvertDatapoint(const DatapointPtr<From>& input,
                                              Datapoint<To>* storage);

template <typename T>
void Scale(const DatapointPtr<T>& input, double scale_factor,
           Datapoint<double>* result);

template <typename T>
DatapointPtr<T> ToDense(const DatapointPtr<T>& input, Datapoint<T>* output);

template <typename T>
DatapointPtr<T> ToSparse(const DatapointPtr<T>& input, Datapoint<T>* output);

template <typename T>
inline void PointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result);
template <typename T>
void SparsePointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result);
template <typename T>
void DensePointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                   Datapoint<T>* result);
template <typename T>
void HybridPointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result);

template <typename T>
inline void WeightedPointSum(const DatapointPtr<T>& a, T a_weight,
                             const DatapointPtr<T>& b, T b_weight,
                             Datapoint<T>* result);

template <typename T>
inline void PointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                      Datapoint<T>* result);
template <typename T>
void SparsePointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result);
template <typename T>
void DensePointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result);
template <typename T>
void HybridPointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result);

template <typename T>
inline void PointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                         Datapoint<T>* result);
template <typename T>
void SparsePointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result);
template <typename T>
void DensePointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result);
template <typename T>
void HybridPointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result);

template <typename T>
inline void PointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result);
template <typename T>
void SparsePointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result);
template <typename T>
void DensePointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                      Datapoint<T>* result);
template <typename T>
void HybridPointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result);

template <typename T>
bool DensePointerIsBinary(const DatapointPtr<T>& pointer);
template <typename T>
bool SparsePointerIsBinary(const DatapointPtr<T>& pointer);
template <typename T>
bool PointerIsBinary(const DatapointPtr<T>& pointer);

template <typename T, typename U>
bool HaveIntersectingNonzeroes(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b) {
  if (a.nonzero_entries() == 0 || b.nonzero_entries() == 0) return false;
  size_t i1_front = 0, i2_front = 0;
  size_t i1_back = a.nonzero_entries() - 1, i2_back = b.nonzero_entries() - 1;

  while (i1_front < i1_back && i2_front < i2_back) {
    if (ABSL_PREDICT_FALSE(a.indices()[i1_front] == b.indices()[i2_front])) {
      return true;
    }

    const size_t to_add_front1 = a.indices()[i1_front] < b.indices()[i2_front];
    const size_t to_add_front2 = a.indices()[i1_front] > b.indices()[i2_front];
    if (ABSL_PREDICT_FALSE(a.indices()[i1_back] == b.indices()[i2_back])) {
      return true;
    }

    const size_t to_sub_back2 = a.indices()[i1_back] < b.indices()[i2_back];
    const size_t to_sub_back1 = a.indices()[i1_back] > b.indices()[i2_back];
    i1_front += to_add_front1;
    i2_front += to_add_front2;
    i1_back -= to_sub_back1;
    i2_back -= to_sub_back2;
  }

  if (i1_front == i1_back) {
    for (; i2_front <= i2_back; ++i2_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[i1_front] == b.indices()[i2_front])) {
        return true;
      }
    }
  } else if (i2_front == i2_back) {
    for (; i1_front <= i1_back; ++i1_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[i1_front] == b.indices()[i2_front])) {
        return true;
      }
    }
  }

  return false;
}

template <typename T>
inline void NormalizeUnitL2(Datapoint<T>* dp) {
  DCHECK(!IsIntegerType<T>());
  dp->MakeNotBinary();
  dp->set_normalization(UNITL2NORM);
  const double l2_norm = SquaredL2Norm(dp->ToPtr());

  if (l2_norm == 0) return;
  const double multiplier = 1.0 / sqrt(l2_norm);
  for (T& elem : *dp->mutable_values()) {
    elem *= multiplier;
  }
}

template <typename To, typename From>
inline void NormalizeUnitL2(const DatapointPtr<From>& input,
                            Datapoint<To>* output) {
  CopyToDatapoint(input, output);
  NormalizeUnitL2(output);
}

template <typename T>
inline Status NormalizeByTag(Normalization tag, Datapoint<T>* dp) {
  DCHECK(dp);
  if (tag == dp->normalization()) return OkStatus();
  switch (tag) {
    case InputOutputConfig::NONE:
      return OkStatus();
    case InputOutputConfig::UNITL2NORM:
      if (IsFloatingType<T>()) {
        NormalizeUnitL2(dp);
        return OkStatus();
      } else {
        return InvalidArgumentError(
            "Cannot normalize a datapoint of integral type such that values "
            "may become non-integral.");
      }
    default:
      LOG(FATAL) << "Normalization type specified by tag not implemented yet.";
  }
}

template <typename T>
double Sum(const DatapointPtr<T>& dptr) {
  return Sum(ConstSpan<T>(dptr.values(), dptr.nonzero_entries()));
}

template <typename T>
void MeanVar(const DatapointPtr<T>& input, double* mean, double* var) {
  const auto dims = input.dimensionality();
  DCHECK_GT(dims, 0) << "Cannot compute MeanVar of zero-dimensional datapoint.";
  DCHECK(mean);
  DCHECK(var);
  double sum_a0 = 0, sum_a1 = 0, sum_aa0 = 0, sum_aa1 = 0;
  auto ptr = input.values();
  auto end = ptr + input.nonzero_entries();
  for (; ptr + 1 < end; ptr += 2) {
    sum_a0 += ptr[0];
    sum_aa0 += ptr[0] * ptr[0];
    sum_a1 += ptr[1];
    sum_aa1 += ptr[1] * ptr[1];
  }

  if (ptr < end) {
    sum_a0 += ptr[0];
    sum_aa0 += ptr[0] * ptr[0];
  }

  sum_a0 += sum_a1;
  sum_aa0 += sum_aa1;
  const double dims_neg_1 = 1.0 / dims;
  *mean = sum_a0 * dims_neg_1;
  *var = (sum_aa0 - *mean * sum_a0) * dims_neg_1;
}

template <typename To, typename From>
inline void CopyToDatapoint(const DatapointPtr<From>& input,
                            Datapoint<To>* output) {
  output->clear();
  output->set_dimensionality(input.dimensionality());
  output->mutable_indices()->insert(output->mutable_indices()->end(),
                                    input.indices_slice().begin(),
                                    input.indices_slice().end());
  output->mutable_values()->insert(output->mutable_values()->end(),
                                   input.values_slice().begin(),
                                   input.values_slice().end());

  constexpr bool need_convert_binary = IsUint8<From>() && !IsUint8<To>();
  if (!need_convert_binary) return;

  if (input.IsSparse() && input.values_slice().empty()) {
    output->mutable_values()->resize(input.nonzero_entries(), To(1));
  }
  if (input.IsDense() && input.nonzero_entries() < input.dimensionality()) {
    DCHECK_EQ(input.nonzero_entries(), DivRoundUp(input.dimensionality(), 8));
    output->mutable_values()->resize(input.dimensionality());
    To* mut_values = output->mutable_values()->data();
    for (size_t j : Seq(input.dimensionality())) {
      mut_values[j] = input.GetElementPacked(j);
    }
  }
}

template <typename T>
void Scale(const DatapointPtr<T>& input, double scale_factor,
           Datapoint<double>* result) {
  result->clear();

  if (input.IsSparse()) {
    result->mutable_indices()->insert(
        result->mutable_indices()->begin(), input.indices(),
        input.indices() + input.nonzero_entries());
    result->set_dimensionality(input.dimensionality());
  }

  result->mutable_values()->reserve(input.nonzero_entries());
  for (size_t i = 0; i < static_cast<size_t>(input.nonzero_entries()); ++i) {
    result->mutable_values()->push_back(input.values()[i] * scale_factor);
  }
}

template <typename To, typename From>
inline DatapointPtr<To> MaybeConvertDatapoint(const DatapointPtr<From>& input,
                                              Datapoint<To>* storage) {
  if (IsSame<To, From>() && !IsUint8<From>()) {
    return *reinterpret_cast<const DatapointPtr<To>*>(&input);
  } else {
    DCHECK(storage);
    CopyToDatapoint(input, storage);
    return storage->ToPtr();
  }
}

template <typename T>
inline DatapointPtr<double> ToDouble(const DatapointPtr<T>& dptr,
                                     Datapoint<double>* dp) {
  return MaybeConvertDatapoint(dptr, dp);
}

template <typename T>
inline DatapointPtr<float> ToFloat(const DatapointPtr<T>& dptr,
                                   Datapoint<float>* dp) {
  return MaybeConvertDatapoint(dptr, dp);
}

template <typename T>
DatapointPtr<T> ToDense(const DatapointPtr<T>& input, Datapoint<T>* output) {
  DCHECK(output);
  if (input.IsDense()) {
    return input;
  } else {
    output->clear();
    auto vals = output->mutable_values();
    vals->resize(input.dimensionality());
    for (size_t i = 0; i < input.nonzero_entries(); ++i) {
      vals->at(input.indices()[i]) = input.values()[i];
    }

    return output->ToPtr();
  }
}

template <typename T>
DatapointPtr<T> ToSparse(const DatapointPtr<T>& input, Datapoint<T>* output) {
  DCHECK(output);
  if (input.IsSparse()) {
    return input;
  } else {
    output->clear();
    output->set_dimensionality(input.dimensionality());
    for (size_t i = 0; i < input.nonzero_entries(); ++i) {
      const T& cur_value = input.values()[i];
      if (cur_value != 0) {
        output->mutable_indices()->push_back(i);
        output->mutable_values()->push_back(cur_value);
      }
    }

    return output->ToPtr();
  }
}

template <typename T>
inline void PointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result) {
  if (a.IsDense()) {
    if (b.IsDense()) {
      return DensePointSum(a, b, result);
    } else {
      return HybridPointSum(a, b, result);
    }
  } else {
    if (b.IsDense()) {
      return HybridPointSum(a, b, result);
    } else {
      return SparsePointSum(a, b, result);
    }
  }
}

template <typename T>
void SparsePointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result) {
  DCHECK(result);
  DCHECK(a.IsSparse());
  DCHECK(b.IsSparse());
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  result->clear();
  result->set_dimensionality(a.dimensionality());

  const DimensionIndex* index_a_ptr = a.indices();
  const DimensionIndex* index_b_ptr = b.indices();
  const DimensionIndex* index_a_end = a.indices() + a.nonzero_entries();
  const DimensionIndex* index_b_end = b.indices() + b.nonzero_entries();
  const T* values_a_ptr = a.values();
  const T* values_b_ptr = b.values();
  if (a.nonzero_entries() > 0 && b.nonzero_entries() > 0) {
    DimensionIndex index_a = *index_a_ptr;
    DimensionIndex index_b = *index_b_ptr;
    while (index_a_ptr < index_a_end && index_b_ptr < index_b_end) {
      if (index_a == index_b) {
        const T sum = *values_a_ptr++ + *values_b_ptr++;
        if (std::abs(sum) != 0) {
          result->mutable_indices()->push_back(index_a);
          result->mutable_values()->push_back(sum);
        }
        ++index_a_ptr;
        ++index_b_ptr;
        if (index_a_ptr != index_a_end) index_a = *index_a_ptr;
        if (index_b_ptr != index_b_end) index_b = *index_b_ptr;
      } else if (index_a < index_b) {
        result->mutable_indices()->push_back(index_a);
        result->mutable_values()->push_back(*values_a_ptr++);
        if (++index_a_ptr == index_a_end) break;
        index_a = *index_a_ptr;
      } else {
        result->mutable_indices()->push_back(index_b);
        result->mutable_values()->push_back(*values_b_ptr++);
        if (++index_b_ptr == index_b_end) break;
        index_b = *index_b_ptr;
      }
    }
  }

  if (index_b_ptr == index_b_end) {
    using std::swap;
    swap(values_a_ptr, values_b_ptr);
    swap(index_a_ptr, index_b_ptr);
    swap(index_a_end, index_b_end);
  }

  for (; index_b_ptr < index_b_end; ++index_b_ptr) {
    result->mutable_indices()->push_back(*index_b_ptr);
    result->mutable_values()->push_back(*values_b_ptr++);
  }
}

template <typename T>
void DensePointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                   Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  result->clear();
  result->mutable_values()->reserve(a.dimensionality());
  for (size_t i = 0; i < b.nonzero_entries(); ++i) {
    result->mutable_values()->push_back(a.values()[i] + b.values()[i]);
  }
}

template <typename T>
void HybridPointSum(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  const DatapointPtr<T>* dense;
  const DatapointPtr<T>* sparse;
  if (a.IsSparse()) {
    DCHECK(b.IsDense());
    dense = &b;
    sparse = &a;
  } else {
    DCHECK(b.IsSparse());
    dense = &a;
    sparse = &b;
  }

  CopyToDatapoint(*dense, result);
  for (size_t i = 0; i < sparse->nonzero_entries(); ++i) {
    (*result->mutable_values())[sparse->indices()[i]] += sparse->values()[i];
  }
}

namespace datapoint_utils_internal {

template <typename T>
void WeightedPointSumDense(const DatapointPtr<T>& a, T a_weight,
                           const DatapointPtr<T>& b, T b_weight,
                           Datapoint<T>* result) {
  const auto d = a.dimensionality();
  DCHECK_EQ(d, b.dimensionality());
  result->set_dimensionality(d);

  auto& values = *result->mutable_values();
  values.resize(d);
  for (DimensionIndex i = 0; i < d; ++i)
    values[i] = a.values()[i] * a_weight + b.values()[i] * b_weight;
}

}  // namespace datapoint_utils_internal

template <typename T>
void WeightedPointSum(const DatapointPtr<T>& a, T a_weight,
                      const DatapointPtr<T>& b, T b_weight,
                      Datapoint<T>* result) {
  DCHECK(result);
  result->clear();

  DCHECK_EQ(a.IsDense(), b.IsDense());
  if (a.IsDense()) {
    datapoint_utils_internal::WeightedPointSumDense(a, a_weight, b, b_weight,
                                                    result);
  } else {
    Datapoint<T> a_buf;
    CopyToDatapoint(a, &a_buf);
    auto& a_buf_values = *a_buf.mutable_values();
    for (DimensionIndex i = 0; i < a_buf.nonzero_entries(); ++i)
      a_buf_values[i] *= a_weight;

    Datapoint<T> b_buf;
    CopyToDatapoint(b, &b_buf);
    auto& b_buf_values = *b_buf.mutable_values();
    for (DimensionIndex i = 0; i < b_buf.nonzero_entries(); ++i)
      b_buf_values[i] *= b_weight;

    PointSum(a_buf.ToPtr(), b_buf.ToPtr(), result);
  }
}

template <typename T>
inline void PointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                         Datapoint<T>* result) {
  if (a.IsDense()) {
    if (b.IsDense()) {
      return DensePointProduct(a, b, result);
    } else {
      return HybridPointProduct(a, b, result);
    }
  } else {
    if (b.IsDense()) {
      return HybridPointProduct(a, b, result);
    } else {
      return SparsePointProduct(a, b, result);
    }
  }
}

template <typename T>
void SparsePointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result) {
  DCHECK(result);
  DCHECK(a.IsSparse());
  DCHECK(b.IsSparse());
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  result->clear();
  result->set_dimensionality(a.dimensionality());

  const DimensionIndex* index_a_ptr = a.indices();
  const DimensionIndex* index_b_ptr = b.indices();
  const DimensionIndex* index_a_end = a.indices() + a.nonzero_entries();
  const DimensionIndex* index_b_end = b.indices() + b.nonzero_entries();
  const T* values_a_ptr = a.values();
  const T* values_b_ptr = b.values();
  if (a.nonzero_entries() > 0 && b.nonzero_entries() > 0) {
    DimensionIndex index_a = *index_a_ptr;
    DimensionIndex index_b = *index_b_ptr;
    while (1) {
      if (index_a == index_b) {
        result->mutable_indices()->push_back(index_a);
        result->mutable_values()->push_back(*values_a_ptr++ * *values_b_ptr++);
        if (++index_a_ptr == index_a_end) break;
        if (++index_b_ptr == index_b_end) break;
        index_a = *index_a_ptr;
        index_b = *index_b_ptr;
      } else if (index_a < index_b) {
        if (++index_a_ptr == index_a_end) break;
        ++values_a_ptr;
        index_a = *index_a_ptr;
      } else {
        if (++index_b_ptr == index_b_end) break;
        ++values_b_ptr;
        index_b = *index_b_ptr;
      }
    }
  }
}

template <typename T>
void DensePointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  result->clear();
  result->mutable_values()->reserve(a.dimensionality());
  for (size_t i = 0; i < b.nonzero_entries(); ++i) {
    result->mutable_values()->push_back(a.values()[i] * b.values()[i]);
  }
}

template <typename T>
void HybridPointProduct(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  const DatapointPtr<T>* dense;
  const DatapointPtr<T>* sparse;
  if (a.IsSparse()) {
    DCHECK(b.IsDense());
    dense = &b;
    sparse = &a;
  } else {
    DCHECK(b.IsSparse());
    dense = &a;
    sparse = &b;
  }

  CopyToDatapoint(*dense, result);
  for (size_t i = 0; i < sparse->nonzero_entries(); ++i) {
    (*result->mutable_values())[sparse->indices()[i]] *= sparse->values()[i];
  }
}

template <typename T>
inline void PointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                        Datapoint<T>* result) {
  if (a.IsDense()) {
    if (b.IsDense()) {
      return DensePointDivide(a, b, result);
    } else {
      return HybridPointDivide(a, b, result);
    }
  } else {
    if (b.IsDense()) {
      return HybridPointDivide(a, b, result);
    } else {
      return SparsePointDivide(a, b, result);
    }
  }
}

template <typename T>
void SparsePointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result) {
  LOG(FATAL) << "Sparse point-by-point division is not feasible.";
}

template <typename T>
void DensePointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                      Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  result->clear();
  result->mutable_values()->reserve(a.dimensionality());
  for (size_t i = 0; i < b.nonzero_entries(); ++i) {
    if (b.values()[i] == 0)
      result->mutable_values()->push_back(0);
    else
      result->mutable_values()->push_back(a.values()[i] / b.values()[i]);
  }
}

template <typename T>
void HybridPointDivide(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                       Datapoint<T>* result) {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  if (a.IsSparse()) {
    DCHECK(b.IsDense());
    result->clear();
    result->mutable_values()->reserve(b.nonzero_entries());
    for (size_t i = 0; i < b.nonzero_entries(); ++i) {
      if (b.values()[i] == 0)
        result->mutable_values()->push_back(0);
      else
        result->mutable_values()->push_back(1.0 / b.values()[i]);
    }

    for (size_t i = 0; i < a.nonzero_entries(); ++i) {
      (*result->mutable_values())[a.indices()[i]] *= a.values()[i];
    }
  } else {
    LOG(FATAL) << "Dense / sparse point division is not feasible.";
  }
}

template <typename T>
inline void PointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                      Datapoint<T>* result) {
  static_assert(IsSignedType<T>(),
                "PointDiff variants cannot be instantiated with unsigned T.");
  if (a.IsDense()) {
    if (b.IsDense()) {
      return DensePointDiff(a, b, result);
    } else {
      return HybridPointDiff(a, b, result);
    }
  } else {
    if (b.IsDense()) {
      return HybridPointDiff(a, b, result);
    } else {
      return SparsePointDiff(a, b, result);
    }
  }
}

template <typename T>
void DensePointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                    Datapoint<T>* result) {
  static_assert(IsSignedType<T>(),
                "PointDiff variants cannot be instantiated with unsigned T.");
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  result->clear();
  result->set_dimensionality(a.dimensionality());
  result->mutable_values()->reserve(a.dimensionality());
  for (size_t i = 0; i < b.nonzero_entries(); ++i) {
    result->mutable_values()->push_back(a.values()[i] - b.values()[i]);
  }
}

template <typename T>
void SparsePointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result) {
  static_assert(IsSignedType<T>(),
                "PointDiff variants cannot be instantiated with unsigned T.");
  DCHECK(result);
  DCHECK(a.IsSparse());
  DCHECK(b.IsSparse());
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  result->clear();
  result->set_dimensionality(a.dimensionality());

  const DimensionIndex* index_a_ptr = a.indices();
  const DimensionIndex* index_b_ptr = b.indices();
  const DimensionIndex* index_a_end = a.indices() + a.nonzero_entries();
  const DimensionIndex* index_b_end = b.indices() + b.nonzero_entries();
  const T* values_a_ptr = a.values();
  const T* values_b_ptr = b.values();
  if (a.nonzero_entries() > 0 && b.nonzero_entries() > 0) {
    DimensionIndex index_a = *index_a_ptr;
    DimensionIndex index_b = *index_b_ptr;
    while (index_a_ptr < index_a_end && index_b_ptr < index_b_end) {
      if (index_a == index_b) {
        const T diff = *values_a_ptr++ - *values_b_ptr++;
        if (std::abs(diff) != 0) {
          result->mutable_indices()->push_back(index_a);
          result->mutable_values()->push_back(diff);
        }
        ++index_a_ptr;
        ++index_b_ptr;
        if (index_a_ptr != index_a_end) index_a = *index_a_ptr;
        if (index_b_ptr != index_b_end) index_b = *index_b_ptr;
      } else if (index_a < index_b) {
        result->mutable_indices()->push_back(index_a);
        result->mutable_values()->push_back(*values_a_ptr++);
        if (++index_a_ptr == index_a_end) break;
        index_a = *index_a_ptr;
      } else {
        result->mutable_indices()->push_back(index_b);
        result->mutable_values()->push_back(-(*values_b_ptr));
        ++values_b_ptr;
        if (++index_b_ptr == index_b_end) break;
        index_b = *index_b_ptr;
      }
    }
  }

  if (index_b_ptr == index_b_end) {
    for (; index_a_ptr < index_a_end; ++index_a_ptr) {
      result->mutable_indices()->push_back(*index_a_ptr);
      result->mutable_values()->push_back(*values_a_ptr++);
    }
  } else {
    for (; index_b_ptr < index_b_end; ++index_b_ptr) {
      result->mutable_indices()->push_back(*index_b_ptr);
      result->mutable_values()->push_back(-*values_b_ptr++);
    }
  }
}

template <typename T>
void HybridPointDiff(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     Datapoint<T>* result) {
  static_assert(IsSignedType<T>(),
                "PointDiff variants cannot be instantiated with unsigned T.");
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  if (a.IsSparse()) {
    DCHECK(b.IsDense());
    result->clear();
    result->mutable_values()->reserve(b.nonzero_entries());
    for (size_t i = 0; i < b.nonzero_entries(); ++i) {
      result->mutable_values()->push_back(-b.values()[i]);
    }

    for (size_t i = 0; i < a.nonzero_entries(); ++i) {
      (*result->mutable_values())[a.indices()[i]] += a.values()[i];
    }
  } else {
    DCHECK(b.IsSparse());
    CopyToDatapoint(a, result);
    for (size_t i = 0; i < b.nonzero_entries(); ++i) {
      (*result->mutable_values())[b.indices()[i]] -= b.values()[i];
    }
  }
}

template <typename T>
bool DensePointerIsBinary(const DatapointPtr<T>& pointer) {
  return std::is_same<T, uint8_t>::value &&
         pointer.nonzero_entries() < pointer.dimensionality();
}

template <typename T>
bool SparsePointerIsBinary(const DatapointPtr<T>& pointer) {
  return std::is_same<T, uint8_t>::value && !pointer.values();
}

template <typename T>
bool PointerIsBinary(const DatapointPtr<T>& pointer) {
  return pointer.IsDense() ? DensePointerIsBinary(pointer)
                           : SparsePointerIsBinary(pointer);
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
