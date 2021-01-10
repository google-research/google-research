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



#ifndef SCANN_DATA_FORMAT_GFV_PROPERTIES_H_
#define SCANN_DATA_FORMAT_GFV_PROPERTIES_H_

#include "scann/data_format/features.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
inline int GfvFeatureType() {
  return GenericFeatureVector::INT64;
}
template <>
inline int GfvFeatureType<float>() {
  return GenericFeatureVector::FLOAT;
}
template <>
inline int GfvFeatureType<double>() {
  return GenericFeatureVector::DOUBLE;
}
template <>
inline int GfvFeatureType<std::string>() {
  return GenericFeatureVector::STRING;
}

string_view GfvFeatureTypeName(int gfv_feature_type);

StatusOr<size_t> GetGfvVectorSize(const GenericFeatureVector& gfv);

StatusOr<DimensionIndex> GetGfvDimensionality(const GenericFeatureVector& gfv);

StatusOr<bool> IsGfvSparse(const GenericFeatureVector& gfv);
StatusOr<bool> IsGfvDense(const GenericFeatureVector& gfv);

inline bool IsNonBinaryNumeric(const GenericFeatureVector& gfv) {
  return gfv.feature_type() == GenericFeatureVector::INT64 ||
         gfv.feature_type() == GenericFeatureVector::FLOAT ||
         gfv.feature_type() == GenericFeatureVector::DOUBLE;
}

Status GetGfvVectorSize(const GenericFeatureVector& gfv,
                        DimensionIndex* result);

Status GetGfvDimensionality(const GenericFeatureVector& gfv,
                            DimensionIndex* result);

Status IsGfvSparse(const GenericFeatureVector& gfv, bool* result);

Status IsGfvDense(const GenericFeatureVector& gfv, bool* result);

size_t GetGfvDimensionalityOrDie(const GenericFeatureVector& gfv);

bool IsGfvSparseOrDie(const GenericFeatureVector& gfv);

bool IsGfvDenseOrDie(const GenericFeatureVector& gfv);

}  // namespace research_scann

#endif
