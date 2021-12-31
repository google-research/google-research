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

#include "scann/partitioning/partitioner_factory.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/partitioning/kmeans_tree_partitioner.pb.h"
#include "scann/partitioning/kmeans_tree_partitioner_utils.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/projection/projection_base.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/types.h"
#include "scann/utils/weak_ptr_cache.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerializedImpl(
    const SerializedPartitioner& proto, const PartitioningConfig& config) {
  const int32_t n_fields_populated =
      proto.has_kmeans() + proto.has_linear_projection();
  if (n_fields_populated != 1) {
    return InvalidArgumentError(
        "SerializedPartitioner must have exactly one subproto field "
        "populated.");
  }

  StatusOr<unique_ptr<Partitioner<T>>> result;
  if (proto.has_kmeans()) {
    auto kmeans_tree =
        std::make_shared<KMeansTree>(proto.kmeans().kmeans_tree());
    return PartitionerFromKMeansTree<T>(std::move(kmeans_tree), config);
  } else if (proto.has_linear_projection()) {
    return InternalError("Linear projection tree partitioners not supported.");
  }

  return InternalError("CAN'T HAPPEN.");
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromSerialized(
    const SerializedPartitioner& proto, const PartitioningConfig& config,
    int32_t projection_seed_offset) {
  if (proto.uses_projection() && !config.has_projection()) {
    return InvalidArgumentError(
        "Serialized partitioner uses projection but PartitioningConfig lacks a "
        "projection subproto.");
  }

  if (!config.has_projection()) {
    return PartitionerFromSerializedImpl<T>(proto, config);
  }

  TF_ASSIGN_OR_RETURN(
      auto projection,
      ProjectionFactory<T>(config.projection(), projection_seed_offset));

  TF_ASSIGN_OR_RETURN(auto partitioner,
                      PartitionerFromSerializedImpl<double>(proto, config));

  return MakeProjectingDecorator<T, double>(std::move(projection),
                                            std::move(partitioner));
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFromKMeansTree(
    shared_ptr<const KMeansTree> kmeans_tree,
    const PartitioningConfig& config) {
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

  auto km = make_unique<KMeansTreePartitioner<T>>(database_tokenization_dist,
                                                  query_tokenization_dist,
                                                  std::move(kmeans_tree));
  km->set_query_spilling_type(config.query_spilling().spilling_type());
  km->set_query_spilling_threshold(
      config.query_spilling().spilling_threshold());
  km->set_query_spilling_type(config.query_spilling().spilling_type());
  km->set_query_spilling_max_centers(
      config.query_spilling().max_spill_centers());

  if (config.database_spilling().spilling_type() ==
      DatabaseSpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    km->set_database_spilling_fixed_number_of_centers(
        config.database_spilling().max_spill_centers());
  }
  if (config.query_tokenization_type() == PartitioningConfig::FLOAT) {
    km->SetQueryTokenizationType(KMeansTreePartitioner<T>::FLOAT);
  } else if (config.query_tokenization_type() ==
             PartitioningConfig::FIXED_POINT_INT8) {
    km->SetQueryTokenizationType(KMeansTreePartitioner<T>::FIXED_POINT_INT8);
  } else if (config.query_tokenization_type() ==
             PartitioningConfig::ASYMMETRIC) {
    SCANN_RETURN_IF_ERROR(
        km->CreateAsymmetricHashingSearcherForQueryTokenization());
    km->SetQueryTokenizationType(
        research_scann::KMeansTreePartitioner<T>::ASYMMETRIC_HASHING);
  }

  if (config.database_tokenization_type() == PartitioningConfig::FLOAT) {
    km->SetDatabaseTokenizationType(KMeansTreePartitioner<T>::FLOAT);
  } else if (config.database_tokenization_type() ==
             PartitioningConfig::FIXED_POINT_INT8) {
    km->SetDatabaseTokenizationType(KMeansTreePartitioner<T>::FIXED_POINT_INT8);
  } else if (config.database_tokenization_type() ==
             PartitioningConfig::ASYMMETRIC) {
    SCANN_RETURN_IF_ERROR(
        km->CreateAsymmetricHashingSearcherForDatabaseTokenization());
    km->SetDatabaseTokenizationType(
        research_scann::KMeansTreePartitioner<T>::ASYMMETRIC_HASHING);
  }

  if (config.compute_residual_stdev()) {
    km->set_populate_residual_stdev(true);
  }

  return StatusOr<unique_ptr<Partitioner<T>>>(std::move(km));
}

SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, int8_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, uint8_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, int16_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, uint16_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, int32_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, uint32_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, int64_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, uint64_t);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, float);
SCANN_INSTANTIATE_SERIALIZED_PARTITIONER_FACTORY(, double);

}  // namespace research_scann
