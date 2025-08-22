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

#include "scann/base/single_machine_factory_scann.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "scann/base/internal/single_machine_factory_impl.h"
#include "scann/base/internal/tree_x_hybrid_factory.h"
#include "scann/base/reordering_helper_factory.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/brute_force/bfloat16_brute_force.h"
#include "scann/brute_force/brute_force.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/partitioning/partitioner_factory.h"
#include "scann/partitioning/partitioner_factory_base.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"
#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/hash_leaf_helpers.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

using std::dynamic_pointer_cast;

namespace research_scann {
namespace {

template <typename T>
StatusOrSearcherUntyped BruteForceFactory(
    const BruteForceConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params) {
  SCANN_RET_CHECK(dataset);

  if (config.fixed_point().enabled() || config.bfloat16().enabled()) {
    return InvalidArgumentError(
        "Quantized brute force only works with float data.");
  }
  auto result = make_unique<BruteForceSearcher<T>>(
      params.pre_reordering_dist, dataset, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);
  result->set_min_distance(params.min_distance);
  return result;
}

StatusOrSearcherUntyped BruteForceFactory(const BruteForceConfig& config,
                                          const GenericSearchParameters& params,
                                          PreQuantizedFixedPoint* fixed_point) {
  shared_ptr<const DenseDataset<int8_t>> fixed_point_dataset =
      std::move(fixed_point->fixed_point_dataset);

  std::vector<float> inverse_multipliers =
      internal::InverseMultiplier(fixed_point);
  shared_ptr<vector<float>> squared_l2_norm_by_datapoint =
      std::move(fixed_point->squared_l2_norm_by_datapoint);
  const auto& distance_type = typeid(*params.reordering_dist);

  if (distance_type == typeid(const DotProductDistance) ||
      distance_type == typeid(const CosineDistance) ||
      distance_type == typeid(const SquaredL2Distance)) {
    auto result = make_unique<ScalarQuantizedBruteForceSearcher>(
        params.reordering_dist, std::move(squared_l2_norm_by_datapoint),
        std::move(fixed_point_dataset),
        make_shared<vector<float>>(std::move(inverse_multipliers)),
        params.pre_reordering_num_neighbors, params.pre_reordering_epsilon);
    result->set_min_distance(params.min_distance);
    return result;
  } else {
    return InvalidArgumentError(
        "Scalar bruteforce is supported only for dot product, cosine "
        "and squared L2 distance.");
  }
}

StatusOrSearcherUntyped BruteForceFactory(
    const BruteForceConfig& config, const GenericSearchParameters& params,
    shared_ptr<DenseDataset<int16_t>> bfloat16_dataset) {
  return make_unique<Bfloat16BruteForceSearcher>(
      params.reordering_dist, std::move(bfloat16_dataset),
      params.pre_reordering_num_neighbors, params.pre_reordering_epsilon,
      config.bfloat16().noise_shaping_threshold());
}

template <>
StatusOrSearcherUntyped BruteForceFactory<float>(
    const BruteForceConfig& config,
    const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params) {
  SCANN_RET_CHECK(dataset);

  if (config.fixed_point().enabled()) {
    const auto tag =
        params.pre_reordering_dist->specially_optimized_distance_tag();
    if (tag != DistanceMeasure::SQUARED_L2 && tag != DistanceMeasure::COSINE &&
        tag != DistanceMeasure::DOT_PRODUCT) {
      return InvalidArgumentError(
          "Scalar-quantized brute force currently only works with "
          "SquaredL2Distance, CosineDistance and DotProductDistance.");
    }
    auto dense = std::dynamic_pointer_cast<DenseDataset<float>>(dataset);
    if (!dense) {
      return InvalidArgumentError(
          "Dataset must be dense for scalar-quantized brute force.");
    }
    if (config.fixed_point().fixed_point_multiplier_quantile() > 1.0f ||
        config.fixed_point().fixed_point_multiplier_quantile() <= 0.0f) {
      return InvalidArgumentError(
          "scalar_quantization_multiplier_quantile must be in (0, 1].");
    }
    ScalarQuantizedBruteForceSearcher::Options opts;
    opts.multiplier_quantile =
        config.fixed_point().fixed_point_multiplier_quantile();
    opts.noise_shaping_threshold =
        config.scalar_quantization_noise_shaping_threshold();
    auto result = make_unique<ScalarQuantizedBruteForceSearcher>(
        params.pre_reordering_dist, dense, params.pre_reordering_num_neighbors,
        params.pre_reordering_epsilon, opts);
    result->set_min_distance(params.min_distance);
    return result;
  } else if (config.bfloat16().enabled()) {
    auto dense = std::dynamic_pointer_cast<DenseDataset<float>>(dataset);
    if (!dense) {
      return InvalidArgumentError(
          "Dataset must be dense for bfloat16 brute force.");
    }
    return make_unique<Bfloat16BruteForceSearcher>(
        params.pre_reordering_dist, dense, params.pre_reordering_num_neighbors,
        params.pre_reordering_epsilon,
        config.bfloat16().noise_shaping_threshold());
  } else {
    auto result = make_unique<BruteForceSearcher<float>>(
        params.pre_reordering_dist, dataset,
        params.pre_reordering_num_neighbors, params.pre_reordering_epsilon);
    result->set_min_distance(params.min_distance);
    return result;
  }
}

template <typename T>
StatusOrSearcherUntyped HashFactory(shared_ptr<TypedDataset<T>> dataset,
                                    const ScannConfig& config,
                                    SingleMachineFactoryOptions* opts,
                                    const GenericSearchParameters& params) {
  const HashConfig& hash_config = config.hash();
  const int num_hashes =
      hash_config.has_asymmetric_hash() + hash_config.has_bit_sampling_hash() +
      hash_config.has_min_hash() + hash_config.has_pca_hash();

  if (num_hashes != 1) {
    return InvalidArgumentError(
        "Exactly one hash type must be configured in HashConfig if using "
        "SingleMachineFactory.");
  }

  if (hash_config.has_asymmetric_hash()) {
    return internal::AsymmetricHasherFactory(dataset, config, opts, params);
  } else {
    return InvalidArgumentError(
        "Asymmetric hashing is the only supported hash type.");
  }
}

class ScannLeafSearcher {
 public:
  template <typename T>
  static StatusOrSearcherUntyped SingleMachineFactoryLeafSearcher(
      const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
      const GenericSearchParameters& params,
      SingleMachineFactoryOptions* opts) {
    if (internal::NumQueryDatabaseSearchTypesConfigured(config) != 1) {
      return InvalidArgumentError(
          "Exactly one single-machine search type must be configured in "
          "ScannConfig if using SingleMachineFactory.");
    }

    if (config.has_partitioning()) {
      return TreeXHybridFactory<T>(
          config, dataset, params,
          &internal::SingleMachineFactoryLeafSearcherScann<T>, opts);
    } else if (config.has_brute_force()) {
      if (std::is_same_v<T, float> &&
          config.brute_force().fixed_point().enabled() &&
          opts->pre_quantized_fixed_point) {
        return BruteForceFactory(config.brute_force(), params,
                                 opts->pre_quantized_fixed_point.get());
      } else if (std::is_same_v<T, float> &&
                 config.brute_force().bfloat16().enabled() &&
                 opts->bfloat16_dataset) {
        return BruteForceFactory(config.brute_force(), params,
                                 opts->bfloat16_dataset);
      } else {
        return BruteForceFactory(config.brute_force(), dataset, params);
      }
    } else if (config.has_hash()) {
      return HashFactory<T>(dataset, config, opts, params);
    } else {
      return UnknownError("Unhandled case");
    }
  }
};

}  // namespace

template <typename T>
StatusOr<unique_ptr<SingleMachineSearcherBase<T>>> SingleMachineFactoryScann(
    const ScannConfig& config, shared_ptr<TypedDataset<T>> dataset,
    SingleMachineFactoryOptions opts) {
  opts.type_tag = TagForType<T>();
  SCANN_ASSIGN_OR_RETURN(auto searcher, SingleMachineFactoryUntypedScann(
                                            config, dataset, std::move(opts)));
  return {
      unique_cast_unsafe<SingleMachineSearcherBase<T>>(std::move(searcher))};
}

StatusOrSearcherUntyped SingleMachineFactoryUntypedScann(
    const ScannConfig& config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts) {
  return internal::SingleMachineFactoryUntypedImpl<ScannLeafSearcher>(
      config, dataset, opts);
}

namespace internal {

template <typename T>
StatusOrSearcherUntyped SingleMachineFactoryLeafSearcherScann(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  return ScannLeafSearcher::SingleMachineFactoryLeafSearcher(config, dataset,
                                                             params, opts);
}

}  // namespace internal

SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN();

}  // namespace research_scann
