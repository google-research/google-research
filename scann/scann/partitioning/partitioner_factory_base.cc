// Copyright 2025 The Google Research Authors.
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

#include "scann/partitioning/partitioner_factory_base.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/random/reservoir_sampling.h"
#include "scann/utils/types.h"

namespace research_scann {

namespace {

size_t ComputeSampleSize(const PartitioningConfig& config,
                         const Dataset* dataset) {
  return config.has_expected_sample_size()
             ? std::min<size_t>(config.expected_sample_size(), dataset->size())
             : static_cast<size_t>(std::ceil(

                   static_cast<double>(
                       config.partitioning_sampling_fraction()) *
                   dataset->size()));
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryNoProjection(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool) {
  const TypedDataset<T>* sampled;
  unique_ptr<TypedDataset<T>> sampled_mutable;

  const size_t sample_size = ComputeSampleSize(config, dataset);
  if (sample_size < dataset->size()) {
    sampled_mutable.reset(
        (dataset->IsSparse())
            ? absl::implicit_cast<TypedDataset<T>*>(new SparseDataset<T>)
            : absl::implicit_cast<TypedDataset<T>*>(new DenseDataset<T>));
    SCANN_RETURN_IF_ERROR(
        sampled_mutable->NormalizeByTag(dataset->normalization()));
    sampled = sampled_mutable.get();
    MTRandom rng(kDeterministicSeed + 1);
    auto sample = ReservoirSampleIdxs(rng, dataset->size(), sample_size);

    sampled_mutable->Reserve(sample.size());
    for (DatapointIndex i : sample) {
      sampled_mutable->AppendOrDie(dataset->at(i), "");
    }
  } else {
    sampled = dataset;
  }
  SCANN_RET_CHECK_EQ(sampled->size(), sample_size);
  LOG(INFO) << "Size of sampled dataset for training partition: "
            << sampled->size();

  return PartitionerFactoryPreSampledAndProjected(sampled, config, pool);
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryWithProjection(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool) {
  const TypedDataset<float>* sampled;
  unique_ptr<TypedDataset<float>> sampled_mutable;
  MTRandom rng(kDeterministicSeed + 1);
  const size_t sample_size = ComputeSampleSize(config, dataset);
  auto sample = ReservoirSampleIdxs(rng, dataset->size(), sample_size);

  SCANN_RET_CHECK(!sample.empty())
      << "Cannot create a partitioner from an empty sampled dataset.";
  bool projected_is_sparse = false;
  DimensionIndex projected_dimensionality = 0;
  SCANN_ASSIGN_OR_RETURN(
      unique_ptr<Projection<T>> projection,
      ProjectionFactory<T>(config.projection(), dataset, 0, pool.get()));
  {
    Datapoint<float> projected;
    SCANN_RETURN_IF_ERROR(
        projection->ProjectInput(dataset->at(sample[0]), &projected));
    projected_is_sparse = projected.IsSparse();

    projected_dimensionality = projected.dimensionality();
  }

  if (projected_is_sparse) {
    sampled_mutable = make_unique<SparseDataset<float>>();
    SCANN_RETURN_IF_ERROR(
        sampled_mutable->NormalizeByTag(dataset->normalization()));
    sampled_mutable->Reserve(sample.size());
    sampled = sampled_mutable.get();
    Datapoint<float> projected;
    for (DatapointIndex i : sample) {
      SCANN_RETURN_IF_ERROR(
          projection->ProjectInput(dataset->at(i), &projected));
      SCANN_RETURN_IF_ERROR(sampled_mutable->Append(projected.ToPtr(), ""));
    }
  } else {
    vector<float> sampled_storage(sample.size() * projected_dimensionality);
    SCANN_RETURN_IF_ERROR(ParallelForWithStatus<1>(
        IndicesOf(sample), pool.get(), [&](size_t sample_idx) -> Status {
          Datapoint<float> projected;
          SCANN_RETURN_IF_ERROR(projection->ProjectInput(
              dataset->at(sample[sample_idx]), &projected));
          std::copy(
              projected.values().begin(), projected.values().end(),
              sampled_storage.begin() + sample_idx * projected_dimensionality);
          return OkStatus();
        }));
    sampled_mutable = make_unique<DenseDataset<float>>(
        std::move(sampled_storage), sample.size());
    SCANN_RETURN_IF_ERROR(
        sampled_mutable->NormalizeByTag(dataset->normalization()));
    sampled = sampled_mutable.get();
  }

  LOG(INFO) << "Size of sampled dataset for training partition: "
            << sampled->size();
  SCANN_ASSIGN_OR_RETURN(
      auto raw_partitioner,
      PartitionerFactoryPreSampledAndProjected(sampled, config, pool));
  return MakeProjectingDecorator<T>(std::move(projection),
                                    std::move(raw_partitioner));
}
}  // namespace

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactory(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool) {
  auto fp = (config.has_projection()) ? (&PartitionerFactoryWithProjection<T>)
                                      : (&PartitionerFactoryNoProjection<T>);
  return (*fp)(dataset, config, pool);
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryPreSampledAndProjected(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> training_parallelization_pool) {
  if (config.tree_type() == PartitioningConfig::KMEANS_TREE) {
    return KMeansTreePartitionerFactoryPreSampledAndProjected(
        dataset, config, training_parallelization_pool);
  } else {
    return InvalidArgumentError("Invalid partitioner type.");
  }
}

SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int16_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int64_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, float);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, double);

}  // namespace research_scann
