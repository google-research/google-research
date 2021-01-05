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

#include "scann/utils/hash_leaf_helpers.h"

#include "absl/synchronization/mutex.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace internal {
namespace {

template <typename T>
using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

StatusOr<shared_ptr<const DistanceMeasure>>
CreateOrGetAymmetricHashingQuantizationDistance(
    const AsymmetricHasherConfig& ah_config,
    const GenericSearchParameters& params) {
  if (ah_config.has_quantization_distance()) {
    return GetDistanceMeasure(ah_config.quantization_distance());
  } else {
    return params.pre_reordering_dist;
  }
}

template <typename T, typename Lambda>
shared_ptr<DenseDataset<uint8_t>> IndexDatabase(
    const TypedDataset<T>& dataset, Lambda index_datapoint_fn,
    shared_ptr<thread::ThreadPool> pool) {
  constexpr size_t kIndexingBlockSize = 128;
  vector<Datapoint<uint8_t>> hashed_vec(dataset.size());
  Status status = OkStatus();
  absl::Mutex status_mutex;
  ParallelFor<kIndexingBlockSize>(
      Seq(dataset.size()), pool.get(), [&](size_t i) {
        Status iter_status = index_datapoint_fn(dataset[i], &hashed_vec[i]);
        if (!iter_status.ok()) {
          absl::MutexLock lock(&status_mutex);
          status = iter_status;
          return;
        }
      });
  if (!status.ok()) {
    LOG(WARNING) << status;
    return nullptr;
  }

  auto mutable_hashed = std::make_shared<DenseDataset<uint8_t>>();

  if (!hashed_vec.empty() &&
      hashed_vec[0].dimensionality() > hashed_vec[0].nonzero_entries()) {
    mutable_hashed->set_packing_strategy(HashedItem::NIBBLE);
    mutable_hashed->set_dimensionality(hashed_vec[0].dimensionality());
  }

  mutable_hashed->Reserve(dataset.size());
  for (size_t i = 0; i < dataset.size(); ++i) {
    mutable_hashed->AppendOrDie(hashed_vec[i].ToPtr(), dataset.GetDocid(i));
    FreeBackingStorage(&hashed_vec[i]);
  }

  return mutable_hashed;
}

}  // namespace

template <typename T>
StatusOr<TrainedAsymmetricHashingResults<T>>
HashLeafHelpers<T>::TrainAsymmetricHashingModel(
    shared_ptr<TypedDataset<T>> dataset, const AsymmetricHasherConfig& config,
    const GenericSearchParameters& params,
    shared_ptr<thread::ThreadPool> pool) {
  if (params.pre_reordering_dist == nullptr) {
    return InvalidArgumentError(
        "pre_reordering_dist in GenericSearchParameters is not "
        "set.");
  }
  TF_ASSIGN_OR_RETURN(
      auto quantization_distance,
      CreateOrGetAymmetricHashingQuantizationDistance(config, params));
  asymmetric_hashing2::TrainingOptions<T> opts(config, quantization_distance,
                                               *dataset);
  TF_ASSIGN_OR_RETURN(
      shared_ptr<const asymmetric_hashing2::Model<T>> model,
      asymmetric_hashing2::TrainSingleMachine<T>(*dataset, opts, pool));
  internal::TrainedAsymmetricHashingResults<T> result;
  result.indexer = std::make_shared<asymmetric_hashing2::Indexer<T>>(
      opts.projector(), quantization_distance, model);
  result.queryer = std::make_shared<asymmetric_hashing2::AsymmetricQueryer<T>>(
      opts.projector(), params.pre_reordering_dist, model);
  result.lookup_type = config.lookup_type();
  result.fixed_point_lut_conversion_options =
      config.fixed_point_lut_conversion_options();
  result.noise_shaping_threshold = config.noise_shaping_threshold();
  if (config.has_centers_filename()) {
    return InvalidArgumentError("Centers file not supported.");
  }
  return result;
}

template <typename T>
StatusOrSearcher<T> HashLeafHelpers<T>::AsymmetricHasherFactory(
    shared_ptr<TypedDataset<T>> dataset,
    shared_ptr<DenseDataset<uint8_t>> hashed_dataset,
    const TrainedAsymmetricHashingResults<T>& training_results,
    const GenericSearchParameters& params,
    shared_ptr<thread::ThreadPool> pool) {
  if (!hashed_dataset) {
    if (std::isnan(training_results.noise_shaping_threshold)) {
      hashed_dataset = IndexDatabase<T>(
          *dataset,
          [&](const DatapointPtr<T>& dptr, Datapoint<uint8_t>* dp) {
            return training_results.indexer->Hash(dptr, dp);
          },
          pool);
    } else {
      hashed_dataset = IndexDatabase<T>(
          *dataset,
          [&](const DatapointPtr<T>& dptr, Datapoint<uint8_t>* dp) {
            return training_results.indexer->HashWithNoiseShaping(
                dptr, dp, training_results.noise_shaping_threshold);
          },
          pool);
    }
    if (!hashed_dataset) {
      return UnknownError("Could not index database.");
    }
  }

  asymmetric_hashing2::SearcherOptions<T> opts;
  opts.set_asymmetric_lookup_type(training_results.lookup_type);
  opts.set_noise_shaping_threshold(training_results.noise_shaping_threshold);
  opts.EnableAsymmetricQuerying(training_results.queryer,
                                training_results.indexer);
  opts.set_fixed_point_lut_conversion_options(
      training_results.fixed_point_lut_conversion_options);
  return StatusOrSearcher<T>(make_unique<asymmetric_hashing2::Searcher<T>>(
      std::move(dataset), std::move(hashed_dataset), std::move(opts),
      params.pre_reordering_num_neighbors, params.pre_reordering_epsilon));
}

template <typename T>
StatusOr<TrainedAsymmetricHashingResults<T>>
HashLeafHelpers<T>::LoadAsymmetricHashingModel(
    const AsymmetricHasherConfig& config, const GenericSearchParameters& params,
    shared_ptr<thread::ThreadPool> pool,
    CentersForAllSubspaces* preloaded_codebook) {
  TF_ASSIGN_OR_RETURN(
      auto quantization_distance,
      CreateOrGetAymmetricHashingQuantizationDistance(config, params));
  shared_ptr<const asymmetric_hashing2::Model<T>> model;
  if (preloaded_codebook) {
    TF_ASSIGN_OR_RETURN(
        model, asymmetric_hashing2::Model<T>::FromProto(*preloaded_codebook));
  } else {
    return InvalidArgumentError("Centers files are not supported.");
  }

  TF_ASSIGN_OR_RETURN(shared_ptr<const ChunkingProjection<T>> projector,
                      ChunkingProjectionFactory<T>(config.projection()));
  internal::TrainedAsymmetricHashingResults<T> result;
  result.indexer = std::make_shared<asymmetric_hashing2::Indexer<T>>(
      projector, quantization_distance, model);
  result.queryer = std::make_shared<asymmetric_hashing2::AsymmetricQueryer<T>>(
      projector, params.pre_reordering_dist, model);
  result.lookup_type = config.lookup_type();
  result.fixed_point_lut_conversion_options =
      config.fixed_point_lut_conversion_options();
  result.noise_shaping_threshold = config.noise_shaping_threshold();
  return result;
}

SCANN_INSTANTIATE_TYPED_CLASS(, TrainedAsymmetricHashingResults);
SCANN_INSTANTIATE_TYPED_CLASS(, HashLeafHelpers);

}  // namespace internal
}  // namespace scann_ops
}  // namespace tensorflow
