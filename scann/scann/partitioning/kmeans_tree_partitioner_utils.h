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

#ifndef SCANN__PARTITIONING_KMEANS_TREE_PARTITIONER_UTILS_H_
#define SCANN__PARTITIONING_KMEANS_TREE_PARTITIONER_UTILS_H_

#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/partitioning/partitioner_base.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>>
KMeansTreePartitionerFactoryPreSampledAndProjected(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<thread::ThreadPool> training_parallelization_pool) {
  DCHECK(dataset);
  const absl::Time start = absl::Now();

  TF_ASSIGN_OR_RETURN(auto training_dist,
                      GetDistanceMeasure(config.partitioning_distance()));

  shared_ptr<const DistanceMeasure> database_tokenization_dist;
  if (config.has_database_tokenization_distance_override()) {
    TF_ASSIGN_OR_RETURN(
        database_tokenization_dist,
        GetDistanceMeasure(config.database_tokenization_distance_override()));
  } else {
    database_tokenization_dist = training_dist;
  }

  shared_ptr<const DistanceMeasure> query_tokenization_dist;
  if (config.has_query_tokenization_distance_override()) {
    TF_ASSIGN_OR_RETURN(
        query_tokenization_dist,
        GetDistanceMeasure(config.query_tokenization_distance_override()));
  } else {
    query_tokenization_dist = training_dist;
  }

  if ((database_tokenization_dist->NormalizationRequired() == UNITL2NORM ||
       query_tokenization_dist->NormalizationRequired() == UNITL2NORM ||
       training_dist->NormalizationRequired() == UNITL2NORM) &&
      config.partitioning_type() == PartitioningConfig::GENERIC) {
    return InvalidArgumentError(
        "Partitioning/tokenization distance measure requires unit L2 "
        "normalization but generic, not spherical, partitioning was selected.");
  }

  auto result = make_unique<KMeansTreePartitioner<T>>(
      database_tokenization_dist, query_tokenization_dist);
  KMeansTreeTrainingOptions opts(config);
  opts.training_parallelization_pool = training_parallelization_pool;
  SCANN_RETURN_IF_ERROR(result->CreatePartitioning(
      *dataset, *training_dist, config.num_children(), &opts));

  result->set_query_spilling_type(config.query_spilling().spilling_type());
  result->set_query_spilling_threshold(
      config.query_spilling().spilling_threshold());
  result->set_query_spilling_type(config.query_spilling().spilling_type());
  result->set_query_spilling_max_centers(
      config.query_spilling().max_spill_centers());
  if (config.database_spilling().spilling_type() ==
      DatabaseSpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    result->set_database_spilling_fixed_number_of_centers(
        config.database_spilling().max_spill_centers());
  }

  if (config.query_tokenization_type() == PartitioningConfig::FLOAT) {
    result->SetQueryTokenizationType(KMeansTreePartitioner<T>::FLOAT);
  } else if (config.query_tokenization_type() ==
             PartitioningConfig::FIXED_POINT_INT8) {
    result->SetQueryTokenizationType(
        KMeansTreePartitioner<T>::FIXED_POINT_INT8);
  }

  if (config.database_tokenization_type() == PartitioningConfig::FLOAT) {
    result->SetDatabaseTokenizationType(KMeansTreePartitioner<T>::FLOAT);
  } else if (config.database_tokenization_type() ==
             PartitioningConfig::FIXED_POINT_INT8) {
    result->SetDatabaseTokenizationType(
        KMeansTreePartitioner<T>::FIXED_POINT_INT8);
  }

  if (config.compute_residual_stdev()) {
    result->set_populate_residual_stdev(true);
  }

  const absl::Time stop = absl::Now();
  LOG(INFO) << "PartitionerFactory ran in " << stop - start << ".";
  return StatusOr<unique_ptr<Partitioner<T>>>(std::move(result));
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
