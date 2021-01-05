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



#ifndef SCANN__UTILS_DATASET_SAMPLING_H_
#define SCANN__UTILS_DATASET_SAMPLING_H_

#include <type_traits>

#include "absl/random/random.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/utils/sampled_index_list.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

enum class SubsamplingStrategy { kWithReplacement, kWithoutReplacement };

namespace internal {

constexpr float kMaxSamplingFractionForBitmap = .15;

template <typename Index>
SampledIndexList<Index> SampleWithReplacement(Index population_size,
                                              Index number_samples,
                                              uint32_t seed) {
  MTRandom rng(seed);
  Index zero = 0;
  vector<Index> sampled(number_samples);
  for (Index i = 0; i < number_samples; ++i)
    sampled[i] = absl::Uniform(rng, zero, population_size);
  return SampledIndexList<Index>(std::move(sampled));
}

template <typename Index>
SampledIndexList<Index> SampleWithoutReplacementBitmap(Index population_size,
                                                       Index number_samples,
                                                       uint32_t seed) {
  MTRandom rng(seed);
  Index zero = 0;
  vector<Index> sampled;
  sampled.reserve(number_samples);
  vector<bool> bitmap(population_size, false);
  while (sampled.size() < number_samples) {
    const Index new_sample = absl::Uniform(rng, zero, population_size);
    if (!bitmap[new_sample]) {
      bitmap[new_sample] = true;
      sampled.push_back(new_sample);
    }
  }
  return SampledIndexList<Index>(std::move(sampled));
}

template <typename Index>
SampledIndexList<Index> SampleWithoutReplacementKnuthMethodS(
    Index population_size, Index number_samples, uint32_t seed) {
  MTRandom rng(seed);
  vector<Index> sampled;
  sampled.reserve(number_samples);
  for (Index i = 0; i < population_size && sampled.size() < number_samples;
       ++i) {
    const Index n = population_size - i;
    const Index m = number_samples - sampled.size();
    if (absl::Uniform<double>(rng, 0, n) <= m) sampled.push_back(i);
  }
  return SampledIndexList<Index>(std::move(sampled));
}

template <typename Index>
StatusOr<SampledIndexList<Index>> CreateSampledIndexList(
    Index population_size, uint32_t seed, float fraction,
    Index min_number_samples, Index max_number_samples,
    SubsamplingStrategy strategy) {
  if (population_size <= 0)
    return InvalidArgumentError(absl::StrCat(
        "Sampling population size must be >= 1, but it is given as ",
        population_size));
  if (fraction < 0.0 || fraction > 1.0)
    return InvalidArgumentError(
        absl::StrCat("Sampling fraction=", fraction, " is NOT within [0, 1]"));
  if (min_number_samples > population_size)
    return InvalidArgumentError(
        absl::StrCat("Sampling min_number_samples=", min_number_samples,
                     " is bigger than ", "population size=", population_size));
  if (min_number_samples > max_number_samples)
    return InvalidArgumentError(
        absl::StrCat("Sampling min_number_samples=", min_number_samples,
                     " is bigger than "
                     "max_number_samples=",
                     max_number_samples));
  if (max_number_samples <= 0)
    return InvalidArgumentError(absl::StrCat(
        "Sampling max_number_samples must be >= 1, but it is given as ",
        max_number_samples));

  max_number_samples = std::min(max_number_samples, population_size);
  Index number_by_fraction = population_size * fraction;
  if (number_by_fraction == 0) {
    LOG(WARNING) << "Force to sample 1 element when trying to sample fraction="
                 << fraction << " out of size=" << population_size
                 << " population";
    number_by_fraction = 1;
  }
  const Index number_samples = std::min(
      std::max(number_by_fraction, min_number_samples), max_number_samples);

  if (number_samples == population_size) {
    return SampledIndexList<Index>(0, population_size);
  } else if (strategy == SubsamplingStrategy::kWithReplacement) {
    return SampleWithReplacement(population_size, number_samples, seed);
  } else if (number_samples <=
             population_size * kMaxSamplingFractionForBitmap) {
    return SampleWithoutReplacementBitmap(population_size, number_samples,
                                          seed);
  } else {
    return SampleWithoutReplacementKnuthMethodS(population_size, number_samples,
                                                seed);
  }
}

template <typename T, template <typename> class OutputDataset>
StatusOr<shared_ptr<OutputDataset<T>>> SubsampleDatasetImpl(
    const TypedDataset<T>& dataset, uint32_t seed, float fraction,
    DatapointIndex min_number_samples, DatapointIndex max_number_samples,
    SubsamplingStrategy strategy) {
  TF_ASSIGN_OR_RETURN(auto sampled,
                      internal::CreateSampledIndexList<DatapointIndex>(
                          dataset.size(), seed, fraction, min_number_samples,
                          max_number_samples, strategy));
  auto result = std::make_shared<OutputDataset<T>>();
  DatapointIndex i;
  while (sampled.GetNextIndex(&i))
    result->AppendOrDie(dataset[i], dataset.GetDocid(i));
  return result;
}

}  // namespace internal

template <typename T>
StatusOr<shared_ptr<TypedDataset<T>>> SubsampleDataset(
    const TypedDataset<T>& dataset, uint32_t seed, float fraction,
    DatapointIndex min_number_samples, DatapointIndex max_number_samples,
    SubsamplingStrategy strategy = SubsamplingStrategy::kWithoutReplacement) {
  if (dataset.IsDense())
    return internal::SubsampleDatasetImpl<T, DenseDataset>(
        dataset, seed, fraction, min_number_samples, max_number_samples,
        strategy);
  else
    return internal::SubsampleDatasetImpl<T, SparseDataset>(
        dataset, seed, fraction, min_number_samples, max_number_samples,
        strategy);
}

template <typename T>
StatusOr<shared_ptr<TypedDataset<T>>> SubsampleDataset(
    const TypedDataset<T>& dataset, uint32_t seed, float fraction) {
  return SubsampleDataset(dataset, seed, fraction, 0, dataset.size());
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
