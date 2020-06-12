// Copyright 2020 The Google Research Authors.
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

#include "scann/base/single_machine_factory_no_sparse.h"

#include <functional>
#include <memory>

#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "scann/base/internal/single_machine_factory_impl.h"
#include "scann/base/reordering_helper_factory.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/brute_force/brute_force.h"
#include "scann/brute_force/scalar_quantized_brute_force.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/partitioner_factory.h"
#include "scann/partitioning/partitioner_factory_base.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"
#include "scann/tree_x_hybrid/tree_x_hybrid_smmd.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/hash_leaf_helpers.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

using std::dynamic_pointer_cast;

namespace tensorflow {
namespace scann_ops {
namespace {

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> CreateTreeXPartitioner(
    shared_ptr<const TypedDataset<T>> dataset, const ScannConfig& config,
    SingleMachineFactoryOptions* opts) {
  if (config.partitioning().num_partitioning_epochs() != 1) {
    return InvalidArgumentError(
        "num_partitioning_epochs must be == 1 for tree-X hybrids.");
  }

  unique_ptr<Partitioner<T>> partitioner;
  if (opts->kmeans_tree) {
    return InvalidArgumentError(
        "pre-trained kmeans-tree partitioners are not supported.");
  } else if (opts->serialized_partitioner) {
    TF_ASSIGN_OR_RETURN(
        partitioner, PartitionerFromSerialized<T>(*opts->serialized_partitioner,
                                                  config.partitioning()));
  } else if (!config.partitioning().has_partitioner_prefix() ||
             config.partitioning().partitioning_on_the_fly()) {
    if (!dataset) {
      return InvalidArgumentError(
          "Partitioning_on_the_fly needs original dataset to proceed.");
    }
    TF_ASSIGN_OR_RETURN(
        partitioner, PartitionerFactory<T>(dataset.get(), config.partitioning(),
                                           opts->parallelization_pool));
  } else {
    return InvalidArgumentError("Loading a partitioner is not supported.");
  }
  if (!partitioner) {
    return UnknownError("Error creating partitioner for tree-X hybrids.");
  }
  partitioner->set_tokenization_mode(UntypedPartitioner::DATABASE);
  return std::move(partitioner);
}

template <typename T>
StatusOrSearcherUntyped AsymmetricHasherFactory(
    shared_ptr<TypedDataset<T>> dataset, const ScannConfig& config,
    SingleMachineFactoryOptions* opts, const GenericSearchParameters& params) {
  const auto& ah_config = config.hash().asymmetric_hash();
  shared_ptr<const DistanceMeasure> quantization_distance;
  std::shared_ptr<thread::ThreadPool> pool = opts->parallelization_pool;
  if (ah_config.has_quantization_distance()) {
    TF_ASSIGN_OR_RETURN(quantization_distance,
                        GetDistanceMeasure(ah_config.quantization_distance()));
  } else {
    quantization_distance = params.pre_reordering_dist;
  }

  internal::TrainedAsymmetricHashingResults<T> training_results;
  if (config.hash().asymmetric_hash().has_centers_filename() ||
      opts->ah_codebook.get()) {
    TF_ASSIGN_OR_RETURN(
        training_results,
        internal::HashLeafHelpers<T>::LoadAsymmetricHashingModel(
            ah_config, params, pool, opts->ah_codebook.get()));
  } else {
    if (!dataset) {
      return InvalidArgumentError(
          "Cannot train AH centers because the dataset is null.");
    }

    if (dataset->size() < ah_config.num_clusters_per_block()) {
      return {make_unique<BruteForceSearcher<T>>(
          params.pre_reordering_dist, dataset,
          params.pre_reordering_num_neighbors, params.pre_reordering_epsilon)};
    }

    const int num_workers = (!pool) ? 0 : (pool->NumThreads());
    LOG(INFO) << "Single-machine AH training with dataset size = "
              << dataset->size() << ", " << num_workers + 1 << " thread(s).";

    TF_ASSIGN_OR_RETURN(
        training_results,
        internal::HashLeafHelpers<T>::TrainAsymmetricHashingModel(
            dataset, ah_config, params, pool));
  }
  return internal::HashLeafHelpers<T>::AsymmetricHasherFactory(
      dataset, opts->hashed_dataset, training_results, params, pool);
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
    return AsymmetricHasherFactory(dataset, config, opts, params);
  } else {
    return InvalidArgumentError(
        "Asymmetric hashing is the only supported hash type.");
  }
}

template <typename T>
StatusOrSearcherUntyped TreeAhHybridResidualFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  return InvalidArgumentError(
      "Tree-AH with residual quantization only works with float data.");
}
template <>
StatusOrSearcherUntyped TreeAhHybridResidualFactory<float>(
    const ScannConfig& config, const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  unique_ptr<Partitioner<float>> partitioner;
  if (config.partitioning().has_partitioner_prefix()) {
    return InvalidArgumentError("Loading a partitioner is not supported.");
  } else {
    TF_ASSIGN_OR_RETURN(partitioner,
                        CreateTreeXPartitioner<float>(dataset, config, opts));
  }
  unique_ptr<KMeansTreeLikePartitioner<float>> kmeans_tree_partitioner(
      dynamic_cast<KMeansTreeLikePartitioner<float>*>(partitioner.release()));
  if (!kmeans_tree_partitioner) {
    return InvalidArgumentError(
        "Tree AH with residual quantization only works with KMeans tree as a "
        "partitioner.");
  }

  auto dense = dynamic_pointer_cast<const DenseDataset<float>>(dataset);
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

  vector<std::vector<DatapointIndex>> datapoints_by_token = {};
  if (dataset && dataset->empty()) {
    datapoints_by_token.resize(kmeans_tree_partitioner->n_tokens());
  } else {
    if (opts->datapoints_by_token) {
      datapoints_by_token = std::move(*opts->datapoints_by_token);
    } else if (dense) {
      TF_ASSIGN_OR_RETURN(datapoints_by_token,
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
    if (datapoints_by_token.size() < kmeans_tree_partitioner->n_tokens()) {
      datapoints_by_token.resize(kmeans_tree_partitioner->n_tokens());
    }
  }

  shared_ptr<const asymmetric_hashing2::Model<float>> ah_model;
  if (opts->ah_codebook) {
    TF_ASSIGN_OR_RETURN(ah_model, asymmetric_hashing2::Model<float>::FromProto(
                                      *opts->ah_codebook));
  } else if (config.hash().asymmetric_hash().has_centers_filename()) {
    return InvalidArgumentError("Centers files are not supported.");
  } else if (dense) {
    if (opts->hashed_dataset) {
      return InvalidArgumentError(
          "If a pre-computed hashed database is specified for tree-AH hybrid "
          "then pre-computed AH centers must be specified too.");
    }
    TF_ASSIGN_OR_RETURN(
        auto quantization_distance,
        GetDistanceMeasure(
            config.hash().asymmetric_hash().quantization_distance()));
    TF_ASSIGN_OR_RETURN(
        auto residuals,
        TreeAHHybridResidual::ComputeResiduals(
            *dense, kmeans_tree_partitioner.get(), datapoints_by_token,
            config.hash()
                .asymmetric_hash()
                .use_normalized_residual_quantization()));
    asymmetric_hashing2::TrainingOptions<float> training_opts(
        config.hash().asymmetric_hash(), quantization_distance, residuals);
    TF_ASSIGN_OR_RETURN(
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
  SCANN_RETURN_IF_ERROR(result->BuildLeafSearchers(
      config.hash().asymmetric_hash(), std::move(kmeans_tree_partitioner),
      std::move(ah_model), std::move(datapoints_by_token),
      opts->hashed_dataset.get(), opts->parallelization_pool.get()));
  opts->datapoints_by_token = nullptr;
  return {std::move(result)};
}

std::vector<float> InverseMultiplier(PreQuantizedFixedPoint* fixed_point) {
  std::vector<float> inverse_multipliers;
  inverse_multipliers.resize(fixed_point->multiplier_by_dimension->size());

  for (size_t i : Seq(inverse_multipliers.size())) {
    inverse_multipliers[i] = 1.0f / fixed_point->multiplier_by_dimension->at(i);
  }
  return inverse_multipliers;
}

void PartitonPreQuantizedFixedPoint(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    PreQuantizedFixedPoint* whole_fp,
    vector<DenseDataset<int8_t>>* tokenized_quantized_datasets,
    vector<std::vector<float>>* tokenized_squared_l2_norms) {
  auto& original = *(whole_fp->fixed_point_dataset);
  auto& original_l2 = *(whole_fp->squared_l2_norm_by_datapoint);
  tokenized_quantized_datasets->clear();
  tokenized_quantized_datasets->resize(datapoints_by_token.size());
  tokenized_squared_l2_norms->clear();
  tokenized_squared_l2_norms->resize(datapoints_by_token.size());

  for (size_t token : IndicesOf(datapoints_by_token)) {
    const auto& subset = datapoints_by_token[token];
    auto& tokenized = tokenized_quantized_datasets->at(token);
    tokenized.set_packing_strategy(original.packing_strategy());
    tokenized.set_dimensionality(original.dimensionality());
    tokenized.Reserve(subset.size());

    auto& tokenized_l2 = (*tokenized_squared_l2_norms)[token];
    tokenized_l2.reserve(subset.size());
    for (const DatapointIndex i : subset) {
      tokenized.AppendOrDie(original[i], "");
    }
    if (!original_l2.empty()) {
      for (const DatapointIndex i : subset) {
        tokenized_l2.push_back(original_l2[i]);
      }
    }
    tokenized.set_normalization_tag(original.normalization());
  }
}

StatusOrSearcherUntyped PretrainedSQTreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  TF_ASSIGN_OR_RETURN(unique_ptr<Partitioner<float>> partitioner,
                      CreateTreeXPartitioner<float>(nullptr, config, opts));
  DCHECK(partitioner);

  auto searcher = make_unique<TreeXHybridSMMD<float>>(
      nullptr, nullptr, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);

  vector<std::vector<DatapointIndex>> datapoints_by_token;
  datapoints_by_token = std::move(*(opts->datapoints_by_token));
  if (datapoints_by_token.size() < partitioner->n_tokens()) {
    datapoints_by_token.resize(partitioner->n_tokens());
  }

  shared_ptr<PreQuantizedFixedPoint> fp = opts->pre_quantized_fixed_point;
  auto inverse_multipliers = InverseMultiplier(fp.get());

  auto build_sq_leaf_lambda =
      [&](DenseDataset<int8_t> scalar_quantized_partition,
          std::vector<float> squared_l2_norms)
      -> StatusOr<unique_ptr<SingleMachineSearcherBase<float>>> {
    auto searcher_or_error = ScalarQuantizedBruteForceSearcher::
        CreateFromQuantizedDatasetAndInverseMultipliers(
            params.pre_reordering_dist, std::move(scalar_quantized_partition),
            std::vector<float>(), std::move(squared_l2_norms),
            params.pre_reordering_num_neighbors, params.pre_reordering_epsilon);
    if (!searcher_or_error.ok()) return searcher_or_error.status();
    auto searcher = std::move(searcher_or_error.ValueOrDie());
    return std::unique_ptr<SingleMachineSearcherBase<float>>(
        searcher.release());
  };

  vector<DenseDataset<int8_t>> tokenized_quantized_datasets;
  vector<std::vector<float>> tokenized_squared_l2_norms;
  PartitonPreQuantizedFixedPoint(datapoints_by_token, fp.get(),
                                 &tokenized_quantized_datasets,
                                 &tokenized_squared_l2_norms);

  if (params.pre_reordering_dist->name() == "SquaredL2Distance" &&
      tokenized_squared_l2_norms.empty()) {
    const auto num_tokens = tokenized_quantized_datasets.size();
    tokenized_squared_l2_norms.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
      auto l2_or_error = ScalarQuantizedBruteForceSearcher::
          ComputeSquaredL2NormsFromQuantizedDataset(
              tokenized_quantized_datasets[i], inverse_multipliers);
      SCANN_RETURN_IF_ERROR(l2_or_error.status());
      tokenized_squared_l2_norms.push_back(std::move(l2_or_error.ValueOrDie()));
    }
  }

  SCANN_RETURN_IF_ERROR(
      searcher->BuildPretrainedScalarQuanitzationLeafSearchers(
          std::move(datapoints_by_token),
          std::move(tokenized_quantized_datasets),
          std::move(tokenized_squared_l2_norms), build_sq_leaf_lambda));
  searcher->set_leaf_searcher_optional_parameter_creator(
      std::make_shared<TreeScalarQuantizationPreprocessedQueryCreator>(
          std::move(inverse_multipliers)));

  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  searcher->set_query_tokenizer(std::move(partitioner));

  SCANN_RETURN_IF_ERROR(
      searcher->set_docids(fp->fixed_point_dataset->docids()));
  fp->fixed_point_dataset = nullptr;
  return {std::move(searcher)};
}

template <typename T>
StatusOrSearcherUntyped NonResidualTreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  TF_ASSIGN_OR_RETURN(auto partitioner,
                      CreateTreeXPartitioner<T>(dataset, config, opts));
  DCHECK(partitioner);

  auto result = make_unique<TreeXHybridSMMD<T>>(
      dataset, opts->hashed_dataset, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon);

  bool use_serialized_per_leaf_hashers = false;
  vector<std::vector<DatapointIndex>> datapoints_by_token;
  bool using_pretokenized_database = false;
  if (config.has_input_output() &&
      config.input_output().has_tokenized_database_wildcard()) {
    using_pretokenized_database = true;

    datapoints_by_token = std::move(*(opts->datapoints_by_token));
    if (datapoints_by_token.size() < partitioner->n_tokens()) {
      datapoints_by_token.resize(partitioner->n_tokens());
    }

    use_serialized_per_leaf_hashers =
        config.has_hash() &&
        ((config.hash().has_pca_hash() &&
          config.hash().has_parameters_filename()) ||
         (config.hash().has_asymmetric_hash() &&
          config.hash().asymmetric_hash().has_centers_filename()));
  }

  if (config.hash().has_asymmetric_hash() &&
      !config.hash().asymmetric_hash().use_per_leaf_partition_training()) {
    const auto& ah_config = config.hash().asymmetric_hash();
    internal::TrainedAsymmetricHashingResults<T> training_results;
    if (config.hash().asymmetric_hash().has_centers_filename() ||
        opts->ah_codebook.get()) {
      TF_ASSIGN_OR_RETURN(
          training_results,
          internal::HashLeafHelpers<T>::LoadAsymmetricHashingModel(
              ah_config, params, opts->parallelization_pool,
              opts->ah_codebook.get()));
    } else {
      TF_ASSIGN_OR_RETURN(
          training_results,
          internal::HashLeafHelpers<T>::TrainAsymmetricHashingModel(
              dataset, ah_config, params, opts->parallelization_pool));
    }
    auto leaf_searcher_builder_lambda =
        [&](shared_ptr<TypedDataset<T>> leaf_dataset,
            shared_ptr<DenseDataset<uint8_t>> leaf_hashed_dataset,
            int32_t token) {
          return internal::HashLeafHelpers<T>::AsymmetricHasherFactory(
              leaf_dataset, leaf_hashed_dataset, training_results, params,
              opts->parallelization_pool);
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
        [&](shared_ptr<TypedDataset<T>> leaf_dataset,
            shared_ptr<DenseDataset<uint8_t>> leaf_hashed_dataset,
            int32_t token) -> StatusOrSearcher<T> {
      SingleMachineFactoryOptions leaf_opts;
      leaf_opts.hashed_dataset = leaf_hashed_dataset;
      leaf_opts.parallelization_pool = opts->parallelization_pool;
      TF_ASSIGN_OR_RETURN(auto leaf_searcher,
                          internal::SingleMachineFactoryLeafSearcherNoSparse<T>(
                              spec_config, leaf_dataset, params, &leaf_opts));
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

  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  result->set_query_tokenizer(std::move(partitioner));
  return {std::move(result)};
}

template <typename T>
StatusOrSearcherUntyped TreeXHybridFactory(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  if (config.hash().asymmetric_hash().use_residual_quantization()) {
    return TreeAhHybridResidualFactory<T>(config, dataset, params, opts);
  } else if (std::is_same<T, float>::value &&
             config.brute_force().fixed_point().enabled() &&
             opts->pre_quantized_fixed_point) {
    return PretrainedSQTreeXHybridFactory(config, nullptr, params, opts);
  } else {
    return NonResidualTreeXHybridFactory<T>(config, dataset, params, opts);
  }
}

template <typename T>
StatusOrSearcherUntyped BruteForceFactory(
    const BruteForceConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params) {
  if (config.fixed_point().enabled()) {
    return InvalidArgumentError(
        "Scalar-quantized brute force only works with float data.");
  }
  return {make_unique<BruteForceSearcher<T>>(
      params.pre_reordering_dist, dataset, params.pre_reordering_num_neighbors,
      params.pre_reordering_epsilon)};
}

StatusOrSearcherUntyped BruteForceFactory(const BruteForceConfig& config,
                                          const GenericSearchParameters& params,
                                          PreQuantizedFixedPoint* fixed_point) {
  auto fixed_point_dataset = std::move(*(fixed_point->fixed_point_dataset));

  std::vector<float> inverse_multipliers = InverseMultiplier(fixed_point);
  auto squared_l2_norm_by_datapoint =
      std::move(*fixed_point->squared_l2_norm_by_datapoint);
  const auto& distance_type = typeid(*params.reordering_dist);

  if (distance_type == typeid(const DotProductDistance) ||
      distance_type == typeid(const CosineDistance) ||
      distance_type == typeid(const SquaredL2Distance)) {
    return {make_unique<ScalarQuantizedBruteForceSearcher>(
        params.reordering_dist, std::move(squared_l2_norm_by_datapoint),
        std::move(fixed_point_dataset), std::move(inverse_multipliers),
        params.pre_reordering_num_neighbors, params.pre_reordering_epsilon)};
  } else {
    return InvalidArgumentError(
        "Scalar bruteforce is supported only for dot product, cosine "
        "and squared L2 distance.");
  }
}

template <>
StatusOrSearcherUntyped BruteForceFactory<float>(
    const BruteForceConfig& config,
    const shared_ptr<TypedDataset<float>>& dataset,
    const GenericSearchParameters& params) {
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
    return {make_unique<ScalarQuantizedBruteForceSearcher>(
        params.pre_reordering_dist, dense, params.pre_reordering_num_neighbors,
        params.pre_reordering_epsilon, opts)};
  } else {
    return {make_unique<BruteForceSearcher<float>>(
        params.pre_reordering_dist, dataset,
        params.pre_reordering_num_neighbors, params.pre_reordering_epsilon)};
  }
}

class NoSparseLeafSearcher {
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
      return TreeXHybridFactory<T>(config, dataset, params, opts);
    } else if (config.has_brute_force()) {
      if (std::is_same<T, float>::value &&
          config.brute_force().fixed_point().enabled() &&
          opts->pre_quantized_fixed_point) {
        return BruteForceFactory(config.brute_force(), params,
                                 opts->pre_quantized_fixed_point.get());
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
StatusOr<unique_ptr<SingleMachineSearcherBase<T>>> SingleMachineFactoryNoSparse(
    const ScannConfig& config, shared_ptr<TypedDataset<T>> dataset,
    SingleMachineFactoryOptions opts) {
  opts.type_tag = TagForType<T>();
  TF_ASSIGN_OR_RETURN(auto searcher, SingleMachineFactoryUntypedNoSparse(
                                         config, dataset, std::move(opts)));
  return {
      unique_cast_unsafe<SingleMachineSearcherBase<T>>(std::move(searcher))};
}

StatusOrSearcherUntyped SingleMachineFactoryUntypedNoSparse(
    const ScannConfig& config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts) {
  return internal::SingleMachineFactoryUntypedImpl<NoSparseLeafSearcher>(
      config, dataset, opts);
}

namespace internal {

template <typename T>
StatusOrSearcherUntyped SingleMachineFactoryLeafSearcherNoSparse(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts) {
  return NoSparseLeafSearcher::SingleMachineFactoryLeafSearcher(config, dataset,
                                                                params, opts);
}

}  // namespace internal

SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_NO_SPARSE();

}  // namespace scann_ops
}  // namespace tensorflow
