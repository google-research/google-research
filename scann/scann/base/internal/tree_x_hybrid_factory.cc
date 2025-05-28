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

#include "scann/base/internal/tree_x_hybrid_factory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/base/internal/single_machine_factory_impl.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/partitioning/partitioner_factory.h"
#include "scann/partitioning/partitioner_factory_base.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/partitioning/tree_brute_force_second_level_wrapper.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"
#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/hash_leaf_helpers.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

template <typename T>
using LeafFactoryT =
    std::function<StatusOrPtr<UntypedSingleMachineSearcherBase>(
        const ScannConfig&, const shared_ptr<TypedDataset<T>>&,
        const GenericSearchParameters&, SingleMachineFactoryOptions*)>;

template <typename T>
struct CreateTreeXPartitionerResult {
  unique_ptr<Partitioner<T>> partitioner;
  vector<std::vector<DatapointIndex>> datapoints_by_token;
};

namespace {

template <typename T>
KMeansTreePartitioner<T>* ExtractKMeansTreePartitioner(
    Partitioner<T>* partitioner) {
  auto kmeans_tree_partitioner =
      dynamic_cast<KMeansTreePartitioner<T>*>(partitioner);
  if (kmeans_tree_partitioner) return kmeans_tree_partitioner;
  auto tree_brute_force_second_level_wrapper =
      dynamic_cast<TreeBruteForceSecondLevelWrapper<T>*>(partitioner);
  if (tree_brute_force_second_level_wrapper) {
    LOG(WARNING) << "Found a TreeBruteForceSecondLevelWrapper in "
                    "TokenizeDatabaseWithAvq.  If AVQ is enabled, it will be "
                    "performed after the second level wrapper is created.  "
                    "This may result in suboptimal recall.";
    return ExtractKMeansTreePartitioner(
        tree_brute_force_second_level_wrapper->base());
  }
  return nullptr;
}

template <typename T>
StatusOr<vector<std::vector<DatapointIndex>>> TokenizeDatabaseWithAvq(
    const PartitioningConfig& pconfig, const TypedDataset<T>& dataset,
    Partitioner<T>* partitioner, ThreadPool* parallelization_pool) {
  {
    auto kmeans_tree_partitioner = ExtractKMeansTreePartitioner(partitioner);
    if (kmeans_tree_partitioner) {
      return kmeans_tree_partitioner->TokenizeDatabase(
          dataset, parallelization_pool,
          {.avq_after_primary = true, .avq_eta = pconfig.avq()});
    }
  }
  auto projecting_partitioner =
      dynamic_cast<KMeansTreeProjectingDecorator<T>*>(partitioner);
  if (!projecting_partitioner) {
    return UnimplementedError(
        "AVQ is only defined for KMeans tree partitioners.");
  }
  auto kmeans_tree_partitioner =
      ExtractKMeansTreePartitioner(projecting_partitioner->base_partitioner());
  if (!kmeans_tree_partitioner) {
    return UnimplementedError(
        "AVQ is only defined for KMeans tree partitioners.");
  }

  auto projection = projecting_partitioner->projection();
  const DimensionIndex projected_dimensionality =
      projection->projected_dimensionality();

  vector<float> projected_dataset_storage(size_t{dataset.size()} *
                                          projected_dimensionality);
  SCANN_RETURN_IF_ERROR(ParallelForWithStatus<1>(
      IndicesOf(dataset), parallelization_pool, [&](size_t i) -> Status {
        Datapoint<float> projected_dp;
        SCANN_RETURN_IF_ERROR(
            projection->ProjectInput(dataset[i], &projected_dp));
        SCANN_RET_CHECK_EQ(projected_dp.values().size(),
                           projected_dimensionality)
            << i;
        SCANN_RET_CHECK_LE((i + 1) * projected_dimensionality,
                           projected_dataset_storage.size())
            << i;
        std::copy(
            projected_dp.values().begin(), projected_dp.values().end(),
            projected_dataset_storage.begin() + i * projected_dimensionality);
        return OkStatus();
      }));
  DenseDataset<float> projected_dataset(std::move(projected_dataset_storage),
                                        dataset.size());
  return kmeans_tree_partitioner->TokenizeDatabase(
      projected_dataset, parallelization_pool,
      {.avq_after_primary = true, .avq_eta = pconfig.avq()});
}
}  // namespace

template <typename T>
StatusOr<CreateTreeXPartitionerResult<T>> CreateTreeXPartitioner(
    shared_ptr<const TypedDataset<T>> dataset, const ScannConfig& config,
    SingleMachineFactoryOptions* opts) {
  const PartitioningConfig& pconfig = config.partitioning();
  if (pconfig.num_partitioning_epochs() != 1) {
    return InvalidArgumentError(
        "num_partitioning_epochs must be == 1 for tree-X hybrids.");
  }

  bool should_apply_avq = false;
  unique_ptr<Partitioner<T>> partitioner;
  if (opts->kmeans_tree) {
    return InvalidArgumentError(
        "pre-trained kmeans-tree partitioners are not supported.");
  } else if (opts->serialized_partitioner) {
    SCANN_ASSIGN_OR_RETURN(
        partitioner,
        PartitionerFromSerialized<T>(*opts->serialized_partitioner, pconfig));
  } else if (!pconfig.has_partitioner_prefix() ||
             pconfig.partitioning_on_the_fly()) {
    if (!dataset) {
      return InvalidArgumentError(
          "Partitioning_on_the_fly needs original dataset to proceed.");
    }
    if (opts->datapoints_by_token) {
      return InvalidArgumentError(
          "Cannot use a pretokenized dataset without a precomputed "
          "partitioner.");
    }
    SCANN_ASSIGN_OR_RETURN(partitioner,
                           PartitionerFactory<T>(dataset.get(), pconfig,
                                                 opts->parallelization_pool));
    should_apply_avq = !std::isnan(pconfig.avq());
  } else {
    return InvalidArgumentError("Loading a partitioner is not supported.");
  }
  if (!partitioner) {
    return UnknownError("Error creating partitioner for tree-X hybrids.");
  }
  partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);

  vector<std::vector<DatapointIndex>> token_to_datapoint_index;
  if (should_apply_avq) {
    SCANN_ASSIGN_OR_RETURN(
        token_to_datapoint_index,
        TokenizeDatabaseWithAvq(pconfig, *dataset, partitioner.get(),
                                opts->parallelization_pool.get()));
  } else if (opts->datapoints_by_token) {
    token_to_datapoint_index = std::move(*opts->datapoints_by_token);
  } else {
    SCANN_ASSIGN_OR_RETURN(token_to_datapoint_index,
                           partitioner->TokenizeDatabase(
                               *dataset, opts->parallelization_pool.get()));
  }
  return CreateTreeXPartitionerResult<T>{std::move(partitioner),
                                         std::move(token_to_datapoint_index)};
}

namespace {
template <typename Searcher>
Status SetOverretrievalFactor(const PartitioningConfig& pconfig,
                              Searcher* searcher) {
  if (pconfig.database_spilling().has_overretrieve_factor()) {
    const float overretrieve_factor =
        pconfig.database_spilling().overretrieve_factor();
    if (!(overretrieve_factor >= 1.0f && overretrieve_factor <= 2.0f)) {
      return InvalidArgumentError(
          absl::StrCat("Invalid overretrieve factor: ", overretrieve_factor,
                       " is out of range [1.0, 2.0]."));
    }
    searcher->set_spilling_overretrieve_factor(
        pconfig.database_spilling().overretrieve_factor());
  }
  return OkStatus();
}

}  // namespace

template <typename T>
StatusOrSearcherUntyped TreeAhHybridResidualFactory(
    ScannConfig config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  return InvalidArgumentError(
      "Tree-AH with residual quantization only works with float data.");
}
template <>
StatusOrSearcherUntyped TreeAhHybridResidualFactory<float>(
    ScannConfig config, const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  unique_ptr<Partitioner<float>> partitioner;
  vector<std::vector<DatapointIndex>> datapoints_by_token;
  if (config.partitioning().has_partitioner_prefix()) {
    return InvalidArgumentError("Loading a partitioner is not supported.");
  } else {
    SCANN_ASSIGN_OR_RETURN(
        auto create_tree_x_partitioner_results,
        CreateTreeXPartitioner<float>(dataset, config, opts));
    datapoints_by_token =
        std::move(create_tree_x_partitioner_results.datapoints_by_token);
    partitioner = std::move(create_tree_x_partitioner_results.partitioner);
  }
  unique_ptr<KMeansTreeLikePartitioner<float>> kmeans_tree_partitioner(
      dynamic_cast<KMeansTreeLikePartitioner<float>*>(partitioner.release()));
  if (!kmeans_tree_partitioner) {
    return InvalidArgumentError(
        "Tree AH with residual quantization only works with KMeans tree as a "
        "partitioner.");
  }

  {
    auto& hash_proj_config =
        *config.mutable_hash()->mutable_asymmetric_hash()->mutable_projection();
    if (!hash_proj_config.has_input_dim() &&
        hash_proj_config.variable_blocks().empty() &&
        hash_proj_config.projection_type() !=
            ProjectionConfig::VARIABLE_CHUNK) {
      hash_proj_config.set_input_dim(kmeans_tree_partitioner->kmeans_tree()
                                         ->CenterForToken(0)
                                         .dimensionality());
      LOG(INFO)
          << "input_dim and num_blocks were not explicitly specified in "
             "TreeAhHybridResidualFactory AH config. Setting input_dim to "
          << hash_proj_config.input_dim()
          << " to match the dimensionality of the partitioning centroids.";

      if (hash_proj_config.input_dim() <
          hash_proj_config.num_dims_per_block()) {
        hash_proj_config.set_num_dims_per_block(hash_proj_config.input_dim());
      }
      if (hash_proj_config.input_dim() < hash_proj_config.num_blocks()) {
        hash_proj_config.set_num_blocks(hash_proj_config.input_dim());
      }
    }
  }

  auto dense = std::dynamic_pointer_cast<const DenseDataset<float>>(dataset);
  if (dataset && !dense) {
    return InvalidArgumentError(
        "Tree-AH with residual quantization only works with dense data.");
  }
  if (params.pre_reordering_dist->specially_optimized_distance_tag() !=
      DistanceMeasure::DOT_PRODUCT) {
    return InvalidArgumentError(
        "Tree-AH with residual quantization only works with dot product "
        "distance for now.");
  }
  auto result = make_unique<TreeAHHybridResidual>(
      dense, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);

  if (dataset && dataset->empty()) {
    datapoints_by_token.resize(kmeans_tree_partitioner->n_tokens());
  } else if (datapoints_by_token.empty()) {
    if (opts->datapoints_by_token) {
      datapoints_by_token = std::move(*opts->datapoints_by_token);
    } else if (dense) {
      SCANN_ASSIGN_OR_RETURN(datapoints_by_token,
                             kmeans_tree_partitioner->TokenizeDatabase(
                                 *dense, opts->parallelization_pool.get()));
    } else {
      return InvalidArgumentError(
          "For Tree-AH hybrid with residual quantization, either "
          "database_wildcard or tokenized_database_wildcard must be provided.");
    }
    if (datapoints_by_token.size() > kmeans_tree_partitioner->n_tokens()) {
      return InvalidArgumentError(
          "The pre-tokenization (ie, datapoints_by_token) specifies %d "
          "partitions, versus the kmeans partitioner, which only has %d "
          "partitions",
          datapoints_by_token.size(), kmeans_tree_partitioner->n_tokens());
    }
  }

  if (datapoints_by_token.size() < kmeans_tree_partitioner->n_tokens()) {
    datapoints_by_token.resize(kmeans_tree_partitioner->n_tokens());
  }

  shared_ptr<const asymmetric_hashing2::Model<float>> ah_model;
  if (opts->ah_codebook) {
    SCANN_ASSIGN_OR_RETURN(
        ah_model,
        asymmetric_hashing2::Model<float>::FromProto(
            *opts->ah_codebook, config.hash().asymmetric_hash().projection()));
  } else if (config.hash().asymmetric_hash().has_centers_filename()) {
    return InvalidArgumentError("Centers files are not supported.");
  } else if (dense) {
    if (opts->hashed_dataset) {
      return InvalidArgumentError(
          "If a pre-computed hashed database is specified for tree-AH hybrid "
          "then pre-computed AH centers must be specified too.");
    }
    SCANN_ASSIGN_OR_RETURN(
        auto quantization_distance,
        GetDistanceMeasure(
            config.hash().asymmetric_hash().quantization_distance()));
    SCANN_ASSIGN_OR_RETURN(
        DenseDataset<float> residuals,
        TreeAHHybridResidual::ComputeResiduals(
            *dense, kmeans_tree_partitioner.get(), datapoints_by_token,
            opts->parallelization_pool.get()));
    asymmetric_hashing2::TrainingOptions<float> training_opts(
        config.hash().asymmetric_hash(), quantization_distance, residuals,
        opts->parallelization_pool.get());
    SCANN_RETURN_IF_ERROR(training_opts.Validate());
    SCANN_ASSIGN_OR_RETURN(
        ah_model, asymmetric_hashing2::TrainSingleMachine(
                      residuals, training_opts, opts->parallelization_pool));
  } else {
    return InvalidArgumentError(
        "For Tree-AH hybrid with residual quantization, either "
        "centers_filename or database_wildcard must be provided.");
  }

  if (!dense) {
    DCHECK(opts->hashed_dataset);
    SCANN_RETURN_IF_ERROR(result->set_docids(opts->hashed_dataset->docids()));
  }

  result->set_database_tokenizer(
      absl::WrapUnique(down_cast<KMeansTreeLikePartitioner<float>*>(
          kmeans_tree_partitioner->Clone().release())));
  if (opts->soar_hashed_dataset && !opts->hashed_dataset) {
    return InvalidArgumentError(
        "If a pre-computed soar_hashed_dataset is specified for tree-AH hybrid "
        "then pre-computed hashed_dataset must be specified too.");
  }
  SCANN_RETURN_IF_ERROR(MaybeAddTopLevelPartitioner(kmeans_tree_partitioner,
                                                    config.partitioning()));
  SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
      config.hash().asymmetric_hash(), std::move(kmeans_tree_partitioner),
      std::move(ah_model), std::move(datapoints_by_token),
      {.hashed_dataset = opts->hashed_dataset.get(),
       .soar_hashed_dataset = opts->soar_hashed_dataset.get(),
       .pool = opts->parallelization_pool.get()}));
  opts->datapoints_by_token = nullptr;
  SCANN_RETURN_IF_ERROR(
      SetOverretrievalFactor(config.partitioning(), result.get()));
  result->set_fixed_point_lut_conversion_options(
      config.hash().asymmetric_hash().fixed_point_lut_conversion_options());
  return {std::move(result)};
}

void PartitionPreQuantizedFixedPoint(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    PreQuantizedFixedPoint* whole_fp,
    vector<DenseDataset<int8_t>>* tokenized_quantized_datasets,
    vector<std::vector<float>>* tokenized_squared_l2_norms) {
  auto& original = *(whole_fp->fixed_point_dataset);
  auto& original_l2 = *(whole_fp->squared_l2_norm_by_datapoint);
  tokenized_quantized_datasets->clear();
  tokenized_quantized_datasets->resize(datapoints_by_token.size());
  tokenized_squared_l2_norms->clear();

  if (!original_l2.empty())
    tokenized_squared_l2_norms->resize(datapoints_by_token.size());

  for (size_t token : IndicesOf(datapoints_by_token)) {
    const auto& subset = datapoints_by_token[token];
    auto& tokenized = tokenized_quantized_datasets->at(token);
    tokenized.set_packing_strategy(original.packing_strategy());
    tokenized.set_dimensionality(original.dimensionality());
    tokenized.Reserve(subset.size());

    for (const DatapointIndex i : subset) {
      tokenized.AppendOrDie(original[i], "");
    }
    if (!original_l2.empty()) {
      auto& tokenized_l2 = (*tokenized_squared_l2_norms)[token];
      tokenized_l2.reserve(subset.size());
      for (const DatapointIndex i : subset) {
        tokenized_l2.push_back(original_l2[i]);
      }
    }
    tokenized.set_normalization_tag(original.normalization());
  }
}

StatusOrPtr<TreeXHybridSMMD<float>> PretrainedTreeSQFactoryFromAssets(
    const ScannConfig& config, const GenericSearchParameters& params,
    const vector<std::vector<DatapointIndex>>& datapoints_by_token,
    unique_ptr<Partitioner<float>> partitioner,
    shared_ptr<PreQuantizedFixedPoint> fp_assets) {
  vector<DenseDataset<int8_t>> tokenized_quantized_datasets;
  vector<std::vector<float>> tokenized_squared_l2_norms;
  PartitionPreQuantizedFixedPoint(datapoints_by_token, fp_assets.get(),
                                  &tokenized_quantized_datasets,
                                  &tokenized_squared_l2_norms);

  auto inverse_multipliers = internal::InverseMultiplier(fp_assets.get());

  auto searcher = make_unique<TreeXHybridSMMD<float>>(
      nullptr, nullptr, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);

  auto build_sq_leaf_lambda =
      [&, inverse_multipliers, params](
          DenseDataset<int8_t> scalar_quantized_partition,
          std::vector<float> squared_l2_norms)
      -> StatusOr<unique_ptr<SingleMachineSearcherBase<float>>> {
    auto searcher_or_error = ScalarQuantizedBruteForceSearcher::
        CreateFromQuantizedDatasetAndInverseMultipliers(
            params.pre_reordering_dist, std::move(scalar_quantized_partition),
            inverse_multipliers, std::move(squared_l2_norms),
            params.pre_reordering_num_neighbors, params.pre_reordering_epsilon);
    if (!searcher_or_error.ok()) return searcher_or_error.status();
    auto searcher = std::move(*searcher_or_error);
    return std::unique_ptr<SingleMachineSearcherBase<float>>(
        searcher.release());
  };
  SCANN_RETURN_IF_ERROR(
      searcher->BuildPretrainedScalarQuantizationLeafSearchers(
          std::move(datapoints_by_token),
          std::move(tokenized_quantized_datasets),
          std::move(tokenized_squared_l2_norms), build_sq_leaf_lambda));
  searcher->set_leaf_searcher_optional_parameter_creator(
      std::make_shared<TreeScalarQuantizationPreprocessedQueryCreator>(
          std::move(inverse_multipliers)));

  searcher->set_database_tokenizer(partitioner->Clone());

  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  searcher->set_query_tokenizer(std::move(partitioner));

  SCANN_RETURN_IF_ERROR(
      searcher->set_docids(fp_assets->fixed_point_dataset->docids()));
  fp_assets->fixed_point_dataset = nullptr;
  SCANN_RETURN_IF_ERROR(
      SetOverretrievalFactor(config.partitioning(), searcher.get()));
  return {std::move(searcher)};
}

StatusOrSearcherUntyped PretrainedSQTreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  vector<std::vector<DatapointIndex>> datapoints_by_token;
  unique_ptr<Partitioner<float>> partitioner;
  SCANN_ASSIGN_OR_RETURN(auto create_tree_x_partitioner_results,
                         CreateTreeXPartitioner<float>(nullptr, config, opts));
  datapoints_by_token =
      std::move(create_tree_x_partitioner_results.datapoints_by_token);
  partitioner = std::move(create_tree_x_partitioner_results.partitioner);
  SCANN_RET_CHECK(partitioner);
  SCANN_RETURN_IF_ERROR(
      MaybeAddTopLevelPartitioner(partitioner, config.partitioning()));

  if (datapoints_by_token.size() < partitioner->n_tokens()) {
    datapoints_by_token.resize(partitioner->n_tokens());
  }
  return PretrainedTreeSQFactoryFromAssets(config, params, datapoints_by_token,
                                           std::move(partitioner),
                                           opts->pre_quantized_fixed_point);
}

template <typename T>
StatusOrSearcherUntyped NonResidualTreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, LeafFactoryT<T> leaf_factory,
    SingleMachineFactoryOptions* opts) {
  const bool using_pretokenized_database =
      opts && opts->datapoints_by_token && !opts->datapoints_by_token->empty();
  unique_ptr<Partitioner<T>> partitioner;
  vector<std::vector<DatapointIndex>> datapoints_by_token;
  SCANN_ASSIGN_OR_RETURN(auto create_tree_x_partitioner_results,
                         CreateTreeXPartitioner<T>(dataset, config, opts));
  datapoints_by_token =
      std::move(create_tree_x_partitioner_results.datapoints_by_token);
  partitioner = std::move(create_tree_x_partitioner_results.partitioner);
  SCANN_RET_CHECK(partitioner);
  if (datapoints_by_token.size() < partitioner->n_tokens()) {
    datapoints_by_token.resize(partitioner->n_tokens());
  }

  const bool use_serialized_per_leaf_hashers =
      using_pretokenized_database && config.has_hash() &&
      ((config.hash().has_pca_hash() &&
        config.hash().has_parameters_filename()) ||
       (config.hash().has_asymmetric_hash() &&
        config.hash().asymmetric_hash().has_centers_filename()));

  if constexpr (std::is_same_v<T, float>) {
    if (config.brute_force().fixed_point().enabled()) {
      auto dense = std::dynamic_pointer_cast<DenseDataset<float>>(dataset);
      if (!dense) {
        return InvalidArgumentError(
            "Dataset must be dense for scalar-quantized brute force.");
      }
      auto sq_config = config.brute_force().fixed_point();
      auto sq_result = ScalarQuantizeFloatDataset(
          *dense, sq_config.fixed_point_multiplier_quantile(),
          sq_config.noise_shaping_threshold());
      auto fp_assets = make_shared<PreQuantizedFixedPoint>();
      fp_assets->fixed_point_dataset = make_shared<DenseDataset<int8_t>>(
          std::move(sq_result.quantized_dataset));
      fp_assets->multiplier_by_dimension = make_shared<vector<float>>(
          std::move(sq_result.multiplier_by_dimension));

      fp_assets->squared_l2_norm_by_datapoint = make_shared<vector<float>>();

      if (!using_pretokenized_database) {
        SCANN_ASSIGN_OR_RETURN(datapoints_by_token,
                               partitioner->TokenizeDatabase(
                                   *dataset, opts->parallelization_pool.get()));
      }
      SCANN_RETURN_IF_ERROR(
          MaybeAddTopLevelPartitioner(partitioner, config.partitioning()));
      SCANN_ASSIGN_OR_RETURN(
          auto result,
          PretrainedTreeSQFactoryFromAssets(config, params, datapoints_by_token,
                                            std::move(partitioner), fp_assets));

      result->ReleaseDatasetAndDocids();
      SCANN_RETURN_IF_ERROR(result->set_docids(dense->docids()));
      SCANN_RETURN_IF_ERROR(
          SetOverretrievalFactor(config.partitioning(), result.get()));
      return result;
    }
  }

  auto result = make_unique<TreeXHybridSMMD<T>>(
      dataset, opts->hashed_dataset, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);

  if (config.hash().has_asymmetric_hash() &&
      !config.hash().asymmetric_hash().use_per_leaf_partition_training()) {
    const auto& ah_config = config.hash().asymmetric_hash();
    internal::TrainedAsymmetricHashingResults<T> training_results;
    if (config.hash().asymmetric_hash().has_centers_filename() ||
        opts->ah_codebook.get()) {
      SCANN_ASSIGN_OR_RETURN(
          training_results,
          internal::HashLeafHelpers<T>::LoadAsymmetricHashingModel(
              ah_config, params, opts->parallelization_pool,
              opts->ah_codebook.get()));
    } else {
      SCANN_ASSIGN_OR_RETURN(
          training_results,
          internal::HashLeafHelpers<T>::TrainAsymmetricHashingModel(
              dataset, ah_config, params, opts->parallelization_pool));
    }

    auto leaf_searcher_builder_lambda =
        [params, training_results,
         parallelization_pool = opts->parallelization_pool](
            shared_ptr<TypedDataset<T>> leaf_dataset,
            shared_ptr<DenseDataset<uint8_t>> leaf_hashed_dataset,
            int32_t token) {
          auto parallelization_pool_or_null =
              token == -1 ? nullptr : parallelization_pool;
          return internal::HashLeafHelpers<T>::AsymmetricHasherFactory(
              leaf_dataset, leaf_hashed_dataset, training_results, params,
              parallelization_pool_or_null);
        };

    std::function<StatusOrSearcher<T>(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder = leaf_searcher_builder_lambda;
    if (using_pretokenized_database) {
      SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
          std::move(datapoints_by_token), leaf_searcher_builder));
    } else {
      SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
          *partitioner, leaf_searcher_builder, opts->parallelization_pool));
    }

    result->set_leaf_searcher_optional_parameter_creator(
        make_unique<
            asymmetric_hashing2::PrecomputedAsymmetricLookupTableCreator<T>>(
            training_results.queryer, training_results.lookup_type));
  } else {
    ScannConfig spec_config = config;
    spec_config.clear_partitioning();
    if (use_serialized_per_leaf_hashers) {
      if (config.hash().has_pca_hash()) {
        spec_config.mutable_hash()->clear_parameters_filename();
      } else if (config.hash().has_asymmetric_hash()) {
        spec_config.mutable_hash()
            ->mutable_asymmetric_hash()
            ->clear_centers_filename();
      }
    }

    auto leaf_searcher_builder_lambda =
        [spec_config, params, leaf_factory,
         parallelization_pool = opts->parallelization_pool](
            shared_ptr<TypedDataset<T>> leaf_dataset,
            shared_ptr<DenseDataset<uint8_t>> leaf_hashed_dataset,
            int32_t token) -> StatusOrSearcher<T> {
      SingleMachineFactoryOptions leaf_opts;
      leaf_opts.hashed_dataset = leaf_hashed_dataset;
      leaf_opts.parallelization_pool =
          token == -1 ? nullptr : parallelization_pool;
      SCANN_ASSIGN_OR_RETURN(
          auto leaf_searcher,
          leaf_factory(spec_config, leaf_dataset, params, &leaf_opts));
      return {unique_cast_unsafe<SingleMachineSearcherBase<T>>(
          std::move(leaf_searcher))};
    };

    std::function<StatusOrSearcher<T>(
        shared_ptr<TypedDataset<T>> dataset_partition,
        shared_ptr<DenseDataset<uint8_t>> hashed_dataset_partition,
        int32_t token)>
        leaf_searcher_builder = leaf_searcher_builder_lambda;
    if (using_pretokenized_database) {
      SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
          std::move(datapoints_by_token), leaf_searcher_builder));
    } else {
      SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
          *partitioner, leaf_searcher_builder, opts->parallelization_pool));
    }
  }

  if (config.has_input_output() &&
      config.input_output().has_tokenized_database_wildcard() &&
      config.has_hash() && config.hash().has_pca_hash() &&
      config.hash().has_parameters_filename()) {
    return InvalidArgumentError("Serialized hashers are not supported.");
  }
  if (result->hashed_dataset()) {
    if (opts->hashed_dataset) opts->hashed_dataset.reset();
    result->ReleaseHashedDataset();
  }

  result->set_database_tokenizer(partitioner->Clone());
  SCANN_RETURN_IF_ERROR(
      MaybeAddTopLevelPartitioner(partitioner, config.partitioning()));
  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  result->set_query_tokenizer(std::move(partitioner));
  SCANN_RETURN_IF_ERROR(
      SetOverretrievalFactor(config.partitioning(), result.get()));
  return {std::move(result)};
}

template <typename T>
StatusOrSearcherUntyped TreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, LeafFactoryT<T> leaf_factory,
    SingleMachineFactoryOptions* opts) {
  if (config.hash().asymmetric_hash().use_residual_quantization()) {
    return TreeAhHybridResidualFactory<T>(config, dataset, params, opts);
  } else if (std::is_same<T, float>::value &&
             config.brute_force().fixed_point().enabled() &&
             opts->pre_quantized_fixed_point) {
    return PretrainedSQTreeXHybridFactory(config, nullptr, params, opts);
  } else {
    return NonResidualTreeXHybridFactory<T>(config, dataset, params,
                                            leaf_factory, opts);
  }
}

SCANN_INSTANTIATE_TREE_X_HYBRID_FACTORY();

}  // namespace research_scann
