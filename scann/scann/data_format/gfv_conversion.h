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



#ifndef SCANN_DATA_FORMAT_GFV_CONVERSION_H_
#define SCANN_DATA_FORMAT_GFV_CONVERSION_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "scann/data_format/features.pb.h"
#include "scann/data_format/gfv_properties.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename VecT>
Status GfvValuesToVector(const GenericFeatureVector& gfv, VecT* result);

inline Status GfvValuesToVectorBitPacked(const GenericFeatureVector& gfv,
                                         std::vector<uint8_t>* result);

template <typename IntT>
void UnpackBinaryToInt64(ConstSpan<IntT> packed, size_t num_dimensions,
                         GenericFeatureVector* gfv);

namespace internal {

template <typename DestT, typename SrcT>
Status SafeForStaticCast(SrcT val) {
  if (IsFloatingType<SrcT>() && !std::isfinite(val)) {
    return InvalidArgumentError("%F is not a valid ScaNN value",
                                static_cast<double>(val));
  }

  if (val < numeric_limits<DestT>::lowest() ||
      val > numeric_limits<DestT>::max()) {
    return InvalidArgumentError(
        "Value %g out of range [%g, %g] in conversion from %s to %s",
        static_cast<double>(val),
        static_cast<double>(numeric_limits<DestT>::lowest()),
        static_cast<double>(numeric_limits<DestT>::max()), TypeName<SrcT>(),
        TypeName<DestT>());
  }
  return OkStatus();
}

template <typename DestT, typename SrcT, typename VecT>
Status AppendRangeToVector(ConstSpan<SrcT> span, VecT* result) {
  DCHECK(result);
  for (SrcT val : span) {
    SCANN_RETURN_IF_ERROR(SafeForStaticCast<DestT>(val));
    result->push_back(val);
  }
  return OkStatus();
}

template <typename T, typename VecT>
enable_if_t<IsUint8<T>(), Status> AppendGfvValuesToVectorBitPacked(
    const GenericFeatureVector& gfv, VecT* result) {
  if (gfv.feature_type() != GenericFeatureVector::BINARY) {
    return InvalidArgumentError(
        "gfv.feature_type must be BINARY for "
        "AppendGfvValuesToVectorBitPacked.");
  }

  const size_t old_size = result->size();
  uint8_t shift = 0;
  static const uint8_t bits_per_int = sizeof(T) * 8;

  result->push_back(0);
  auto end_ptr = &result->back();
  for (const int64_t elem : gfv.feature_value_int64()) {
    if (shift == bits_per_int) {
      shift = 0;
      result->push_back(0);
      end_ptr = &result->back();
    }

    switch (elem) {
      case 0:
        break;
      case 1:
        *end_ptr |= static_cast<T>(1) << shift;
        break;
      default:
        result->resize(old_size);
        return InvalidArgumentError(
            "Can only append 0 or 1 to a binary vector, not %d.", elem);
    }
    ++shift;
  }

  return OkStatus();
}

template <typename T, typename VecT>
enable_if_t<!IsUint8<T>(), Status> AppendGfvValuesToVectorBitPacked(
    const GenericFeatureVector& gfv, VecT* result) {
  return AppendRangeToVector<T>(MakeConstSpan(gfv.feature_value_int64()),
                                result);
}

template <typename T, typename VecT>
Status AppendGfvValuesToVector(const GenericFeatureVector& gfv, VecT* result) {
  DCHECK(result);

  switch (gfv.feature_type()) {
    case GenericFeatureVector::INT64:
      return AppendRangeToVector<T>(MakeConstSpan(gfv.feature_value_int64()),
                                    result);

    case GenericFeatureVector::FLOAT:

      return AppendRangeToVector<T>(MakeConstSpan(gfv.feature_value_float()),
                                    result);

    case GenericFeatureVector::DOUBLE:
      return AppendRangeToVector<T>(MakeConstSpan(gfv.feature_value_double()),
                                    result);

    case GenericFeatureVector::BINARY:
      return AppendGfvValuesToVectorBitPacked<T>(gfv, result);

    default:
      return InvalidArgumentError("Feature type not known:  %d",
                                  gfv.feature_type());
  }
}

}  // namespace internal

template <typename VecT>
Status GfvValuesToVector(const GenericFeatureVector& gfv, VecT* result) {
  DCHECK(result);
  TF_ASSIGN_OR_RETURN(size_t to_reserve, GetGfvVectorSize(gfv));
  result->clear();
  result->reserve(to_reserve);
  using T = decay_t<decltype((*result)[0])>;
  return internal::AppendGfvValuesToVector<T>(gfv, result);
}

Status GfvValuesToVectorBitPacked(const GenericFeatureVector& gfv,
                                  std::vector<uint8_t>* result) {
  DCHECK(result);
  result->clear();
  return internal::AppendGfvValuesToVectorBitPacked<uint8_t>(gfv, result);
}

template <typename IntT>
void UnpackBinaryToInt64(ConstSpan<IntT> packed, size_t num_dimensions,
                         GenericFeatureVector* gfv) {
  DCHECK(gfv);
  gfv->Clear();
  gfv->set_feature_type(GenericFeatureVector::BINARY);
  static const IntT int_one = 1;
  size_t dims_converted = 0;
  for (IntT elem : packed) {
    if (dims_converted == num_dimensions) break;
    for (uint8_t shift = 0; shift < sizeof(IntT) * 8; ++shift) {
      const int64_t val = (elem & (int_one << shift)) >> shift;
      gfv->add_feature_value_int64(val);
      if (++dims_converted == num_dimensions) break;
    }
  }
}

template <>
inline void UnpackBinaryToInt64<float>(ConstSpan<float> packed,
                                       size_t num_dimensions,
                                       GenericFeatureVector* gfv) {
  LOG(FATAL) << "Cannot treat float arrays as bit-packed binary data.";
}

template <>
inline void UnpackBinaryToInt64<double>(ConstSpan<double> packed,
                                        size_t num_dimensions,
                                        GenericFeatureVector* gfv) {
  LOG(FATAL) << "Cannot treat double arrays as bit-packed binary data.";
}

}  // namespace research_scann

#endif
