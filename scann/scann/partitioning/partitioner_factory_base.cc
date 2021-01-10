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

#include "scann/partitioning/partitioner_factory_base.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

namespace {

float ComputeSamplingFraction(const PartitioningConfig& config,
                              const Dataset* dataset) {
  return (config.has_expected_sample_size())
             ? std::min(1.0,
                        static_cast<double>(config.expected_sample_size()) /
                            dataset->size())
             : config.partitioning_sampling_fraction();
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryNoProjection(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool) {
  const TypedDataset<T>* sampled;
  unique_ptr<TypedDataset<T>> sampled_mutable;

  const float sampling_fraction = ComputeSamplingFraction(config, dataset);
  if (sampling_fraction < 1.0) {
    sampled_mutable.reset(
        (dataset->IsSparse())
            ? absl::implicit_cast<TypedDataset<T>*>(new SparseDataset<T>)
            : absl::implicit_cast<TypedDataset<T>*>(new DenseDataset<T>));
    SCANN_RETURN_IF_ERROR(
        sampled_mutable->NormalizeByTag(dataset->normalization()));
    sampled = sampled_mutable.get();
    MTRandom rng(kDeterministicSeed + 1);
    vector<DatapointIndex> sample;
    for (DatapointIndex i = 0; i < dataset->size(); ++i) {
      if (absl::Uniform<float>(rng, 0, 1) < sampling_fraction) {
        sample.push_back(i);
      }
    }

    sampled_mutable->Reserve(sample.size());
    for (DatapointIndex i : sample) {
      sampled_mutable->AppendOrDie(dataset->at(i), "");
    }
  } else {
    sampled = dataset;
  }
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
  vector<DatapointIndex> sample;
  const float sampling_fraction = ComputeSamplingFraction(config, dataset);
  for (DatapointIndex i = 0; i < dataset->size(); ++i) {
    if (absl::Uniform<float>(rng, 0, 1) < sampling_fraction) {
      sample.push_back(i);
    }
  }

  auto append_to_sampled = [&](const DatapointPtr<float>& dptr) -> Status {
    if (ABSL_PREDICT_FALSE(!sampled_mutable)) {
      if (dptr.IsSparse()) {
        sampled_mutable = make_unique<SparseDataset<float>>();
      } else {
        sampled_mutable = make_unique<DenseDataset<float>>();
      }
      sampled_mutable->Reserve(sample.size());
      SCANN_RETURN_IF_ERROR(
          sampled_mutable->NormalizeByTag(dataset->normalization()));
      sampled = sampled_mutable.get();
    }
    return sampled_mutable->Append(dptr, "");
  };
  TF_ASSIGN_OR_RETURN(unique_ptr<Projection<T>> projection,
                      ProjectionFactory(config.projection(), dataset));
  Datapoint<float> projected;
  for (DatapointIndex i : sample) {
    SCANN_RETURN_IF_ERROR(projection->ProjectInput(dataset->at(i), &projected));
    SCANN_RETURN_IF_ERROR(append_to_sampled(projected.ToPtr()));
  }
  LOG(INFO) << "Size of sampled dataset for training partition: "
            << sampled->size();
  TF_ASSIGN_OR_RETURN(
      auto raw_partitioner,
      PartitionerFactoryPreSampledAndProjected(sampled, config, pool));
  return MakeProjectingDecorator<T, float>(std::move(projection),
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
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint16_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int64_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint64_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, float);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, double);

}  // namespace research_scann
