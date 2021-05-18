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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_JACCARD_DISTANCE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_JACCARD_DISTANCE_H_

#include <cstdint>

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace research_scann {

class GeneralJaccardDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(NOT_SPECIALLY_OPTIMIZED);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const;

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const;

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    LOG(FATAL) << "Not implemented yet.";
  }
};

class BinaryJaccardDistance final : public BinaryDistanceMeasureBase {
 public:
  string_view name() const final;

  using BinaryDistanceMeasureBase::GetDistanceDense;
  using BinaryDistanceMeasureBase::GetDistanceHybrid;
  using BinaryDistanceMeasureBase::GetDistanceSparse;

  double GetDistanceDense(const DatapointPtr<uint8_t>& a,
                          const DatapointPtr<uint8_t>& b) const final;
  double GetDistanceSparse(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final;
  double GetDistanceHybrid(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final {
    return GetDistanceSparse(a, b);
  }
};

template <typename T>
double GeneralJaccardDistance::GetDistanceDenseImpl(
    const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {
  double intersection = 0.0;
  double sum = 0.0;
  for (size_t i = 0; i < a.dimensionality(); ++i) {
    T en1 = a.values()[i];
    T en2 = b.values()[i];
    intersection += std::min(en1, en2);
    sum += std::max(en1, en2);
  }
  if (!sum) {
    return 0;
  }
  double ratio = intersection / sum;
  return 1.0 - ratio;
}

template <typename T>
double GeneralJaccardDistance::GetDistanceSparseImpl(
    const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {
  double intersection = 0.0;
  double sum = 0.0;
  auto s1 = a.nonzero_entries();
  auto s2 = b.nonzero_entries();
  std::vector<pair<DimensionIndex, double> > w1;
  std::vector<pair<DimensionIndex, double> > w2;
  std::vector<pair<DimensionIndex, double> > merged(s1 + s2);
  w1.reserve(s1);
  for (size_t i = 0; i < s1; ++i) {
    w1.push_back(std::make_pair(a.indices()[i], a.values()[i]));
  }
  w2.reserve(s2);
  for (size_t i = 0; i < s2; ++i) {
    w2.push_back(std::make_pair(b.indices()[i], b.values()[i]));
  }
  std::merge(w1.begin(), w1.end(), w2.begin(), w2.end(), merged.begin());
  while (true) {
    if (merged.empty()) break;
    if (merged.size() == 1) {
      sum += merged[0].second;
      break;
    }
    auto in1 = merged[merged.size() - 1].first;
    auto in2 = merged[merged.size() - 2].first;
    if (in1 == in2) {
      sum += std::max(merged[merged.size() - 1].second,
                      merged[merged.size() - 2].second);
      intersection += std::min(merged[merged.size() - 1].second,
                               merged[merged.size() - 2].second);
      merged.pop_back();
      merged.pop_back();
    } else {
      sum += merged[merged.size() - 1].second;
      merged.pop_back();
    }
  }
  if (!sum) {
    return 0;
  }
  double ratio = intersection / sum;
  return 1.0 - ratio;
}

}  // namespace research_scann

#endif
