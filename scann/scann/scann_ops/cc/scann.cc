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

#include "scann/scann_ops/cc/scann.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/brute_force.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/single_machine_retraining.h"
#include "scann/utils/threads.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace {

constexpr const int32_t kSoarEmptyToken = -1;

bool HasSoar(const ScannConfig& config) {
  return config.partitioning().database_spilling().spilling_type() ==
         DatabaseSpillingConfig::TWO_CENTER_ORTHOGONALITY_AMPLIFIED;
}

int GetNumCPUs() { return std::max(absl::base_internal::NumCPUs(), 1); }

unique_ptr<DenseDataset<float>> InitDataset(
    ConstSpan<float> dataset, DatapointIndex n_points,
    DimensionIndex n_dim = kInvalidDimension) {
  if (dataset.empty() && n_dim == kInvalidDimension) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  auto ds =
      std::make_unique<DenseDataset<float>>(std::move(dataset_vec), n_points);
  if (n_dim != kInvalidDimension) {
    ds->set_dimensionality(n_dim);
  }
  return ds;
}

Status AddTokenizationToOptions(SingleMachineFactoryOptions& opts,
                                ConstSpan<int32_t> tokenization,
                                const int spilling_mult = 1) {
  if (tokenization.empty()) return OkStatus();
  if (opts.serialized_partitioner == nullptr)
    return FailedPreconditionError(
        "Non-empty tokenization but no serialized partitioner is present.");
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          opts.serialized_partitioner->n_tokens());
  for (auto [dp_idx, token] : Enumerate(tokenization)) {
    if (token != kSoarEmptyToken) {
      opts.datapoints_by_token->at(token).push_back(dp_idx / spilling_mult);
    }
  }
  return OkStatus();
}

}  // namespace

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const ScannConfig& config, const ScannAssets& orig_assets) {
  ScannAssets assets = orig_assets;
  SingleMachineFactoryOptions opts;

  std::sort(assets.mutable_assets()->pointer_begin(),
            assets.mutable_assets()->pointer_end(),
            [](const ScannAsset* a, const ScannAsset* b) {
              const auto to_int = [](ScannAsset::AssetType a) -> int {
                if (a == ScannAsset::PARTITIONER) return 0;
                if (a == ScannAsset::TOKENIZATION_NPY) return 1;
                return 2 + a;
              };
              return to_int(a->asset_type()) < to_int(b->asset_type());
            });

  unique_ptr<FixedLengthDocidCollection> docids;

  shared_ptr<DenseDataset<float>> dataset;
  auto fp = make_shared<PreQuantizedFixedPoint>();
  for (const ScannAsset& asset : assets.assets()) {
    const string_view asset_path = asset.asset_path();
    switch (asset.asset_type()) {
      case ScannAsset::AH_CENTERS:
        opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
        SCANN_RETURN_IF_ERROR(
            ReadProtobufFromFile(asset_path, opts.ah_codebook.get()));
        break;
      case ScannAsset::PARTITIONER:
        opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
        SCANN_RETURN_IF_ERROR(ReadProtobufFromFile(
            asset_path, opts.serialized_partitioner.get()));
        break;
      case ScannAsset::TOKENIZATION_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<int32_t>(asset_path));
        const int spilling_mult = HasSoar(config) ? 2 : 1;
        SCANN_RETURN_IF_ERROR(AddTokenizationToOptions(
            opts, vector_and_shape.first, spilling_mult));
        if (HasSoar(config)) {
          docids = std::make_unique<FixedLengthDocidCollection>(4);
          docids->Reserve(vector_and_shape.second[0] / 2);

          for (size_t i = 1; i < vector_and_shape.second[0]; i += 2) {
            int32_t token = vector_and_shape.first[i];
            SCANN_RETURN_IF_ERROR(docids->Append(strings::Int32ToKey(token)));
          }
        }
        break;
      }
      case ScannAsset::AH_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<uint8_t>(asset_path));
        opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      case ScannAsset::AH_DATASET_SOAR_NPY: {
        DCHECK(HasSoar(config));
        DCHECK_NE(docids, nullptr);
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<uint8_t>(asset_path));
        opts.soar_hashed_dataset = make_shared<DenseDataset<uint8_t>>(
            std::move(vector_and_shape.first), std::move(docids));
        break;
      }
      case ScannAsset::DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        dataset = make_shared<DenseDataset<float>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);

        if (vector_and_shape.second[0] == 0)
          dataset->set_dimensionality(vector_and_shape.second[1]);
        break;
      }
      case ScannAsset::INT8_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<int8_t>(asset_path));
        fp->fixed_point_dataset = make_shared<DenseDataset<int8_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);

        if (vector_and_shape.second[0] == 0)
          fp->fixed_point_dataset->set_dimensionality(
              vector_and_shape.second[1]);
        break;
      }
      case ScannAsset::INT8_MULTIPLIERS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        fp->multiplier_by_dimension =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      case ScannAsset::INT8_NORMS_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<float>(asset_path));
        fp->squared_l2_norm_by_datapoint =
            make_shared<vector<float>>(std::move(vector_and_shape.first));
        break;
      }
      case ScannAsset::BF16_DATASET_NPY: {
        SCANN_ASSIGN_OR_RETURN(auto vector_and_shape,
                               NumpyToVectorAndShape<int16_t>(asset_path));
        opts.bfloat16_dataset = make_shared<DenseDataset<int16_t>>(
            std::move(vector_and_shape.first), vector_and_shape.second[0]);
        break;
      }
      default:
        break;
    }
  }
  if (fp->fixed_point_dataset != nullptr) {
    if (fp->squared_l2_norm_by_datapoint == nullptr)
      fp->squared_l2_norm_by_datapoint = make_shared<vector<float>>();
    opts.pre_quantized_fixed_point = fp;
  }
  return std::make_tuple(config, std::move(dataset), std::move(opts));
}

std::string RewriteAssetFilenameIfRelative(const string& artifacts_dir,
                                           const string& asset_path) {
  std::filesystem::path path(asset_path);
  if (path.is_relative()) {
    return (artifacts_dir / path).string();
  } else {
    return asset_path;
  }
}

StatusOr<ScannInterface::ScannArtifacts> ScannInterface::LoadArtifacts(
    const std::string& artifacts_dir, const std::string& scann_assets_pbtxt) {
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(
      ReadProtobufFromFile(artifacts_dir + "/scann_config.pb", &config));
  ScannAssets assets;
  if (scann_assets_pbtxt.empty()) {
    SCANN_ASSIGN_OR_RETURN(auto assets_pbtxt,
                           GetContents(artifacts_dir + "/scann_assets.pbtxt"));
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, assets_pbtxt));
  } else {
    SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  }
  for (auto i : Seq(assets.assets_size())) {
    auto new_path = RewriteAssetFilenameIfRelative(
        artifacts_dir, assets.assets(i).asset_path());
    assets.mutable_assets(i)->set_asset_path(new_path);
  }
  return LoadArtifacts(config, assets);
}

StatusOr<std::unique_ptr<SingleMachineSearcherBase<float>>>
ScannInterface::CreateSearcher(ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);

  if (dataset && config.has_partitioning() &&
      config.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(research_scann::UNITL2NORM);

  SCANN_ASSIGN_OR_RETURN(auto searcher, SingleMachineFactoryScann<float>(
                                            config, dataset, std::move(opts)));
  searcher->MaybeReleaseDataset();
  return searcher;
}

Status ScannInterface::Initialize(const std::string& config_pbtxt,
                                  const std::string& scann_assets_pbtxt) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config_pbtxt));
  ScannAssets assets;
  SCANN_RETURN_IF_ERROR(ParseTextProto(&assets, scann_assets_pbtxt));
  SCANN_ASSIGN_OR_RETURN(auto dataset_and_opts, LoadArtifacts(config_, assets));
  auto [_, dataset, opts] = std::move(dataset_and_opts);
  return Initialize(std::tie(config_, dataset, opts));
}

Status ScannInterface::Initialize(
    ScannConfig config, SingleMachineFactoryOptions opts,
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points) {
  config_ = config;
  if (opts.ah_codebook != nullptr) {
    vector<uint8_t> hashed_db(hashed_dataset.data(),
                              hashed_dataset.data() + hashed_dataset.size());
    opts.hashed_dataset =
        std::make_shared<DenseDataset<uint8_t>>(std::move(hashed_db), n_points);
  }
  const int spilling_mult = HasSoar(config_) ? 2 : 1;
  SCANN_RETURN_IF_ERROR(
      AddTokenizationToOptions(opts, datapoint_to_token, spilling_mult));
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(std::move(int8_vec), n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }

  DimensionIndex n_dim = kInvalidDimension;
  if (config.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config.input_output().pure_dynamic_config().dimensionality();
  return Initialize(std::make_tuple(
      config_, InitDataset(dataset, n_points, n_dim), std::move(opts)));
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  const std::string& config,
                                  int training_threads) {
  SCANN_RETURN_IF_ERROR(ParseTextProto(&config_, config));
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = GetNumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);

  DimensionIndex n_dim = kInvalidDimension;
  if (config_.input_output().pure_dynamic_config().has_dimensionality())
    n_dim = config_.input_output().pure_dynamic_config().dimensionality();
  return Initialize(std::make_tuple(
      config_, InitDataset(dataset, n_points, n_dim), std::move(opts)));
}

Status ScannInterface::Initialize(ScannInterface::ScannArtifacts artifacts) {
  auto [config, dataset, opts] = std::move(artifacts);
  config_ = config;
  SCANN_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                              config_, dataset.get()));
  SCANN_ASSIGN_OR_RETURN(scann_,
                         CreateSearcher(std::tie(config_, dataset, opts)));
  if (scann_->config().has_value()) config_ = scann_->config().value();

  const std::string& distance = config_.distance_measure().distance_measure();
  const absl::flat_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;

  if (config_.has_partitioning()) {
    min_batch_size_ = 1;
  } else {
    if (config_.has_hash())
      min_batch_size_ = 16;
    else
      min_batch_size_ = 256;
  }
  parallel_query_pool_ = StartThreadPool("ScannQueryingPool", GetNumCPUs() - 1);
  return OkStatus();
}

SearchParameters ScannInterface::GetSearchParameters(int final_nn,
                                                     int pre_reorder_nn,
                                                     int leaves) const {
  SearchParameters params;
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  params.set_pre_reordering_num_neighbors(pre_reorder_nn);
  params.set_post_reordering_num_neighbors(post_reorder_nn);
  if (leaves > 0) {
    auto tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
    params.set_searcher_specific_optional_parameters(tree_params);
  }
  return params;
}

vector<SearchParameters> ScannInterface::GetSearchParametersBatched(
    int batch_size, int final_nn, int pre_reorder_nn, int leaves,
    bool set_unspecified) const {
  vector<SearchParameters> params(batch_size);
  bool has_reordering = config_.has_exact_reordering();
  int post_reorder_nn = -1;
  if (has_reordering) {
    post_reorder_nn = final_nn;
  } else {
    pre_reorder_nn = final_nn;
  }
  std::shared_ptr<research_scann::TreeXOptionalParameters> tree_params;
  if (leaves > 0) {
    tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
  }

  for (auto& p : params) {
    p.set_pre_reordering_num_neighbors(pre_reorder_nn);
    p.set_post_reordering_num_neighbors(post_reorder_nn);
    if (tree_params) p.set_searcher_specific_optional_parameters(tree_params);
    if (set_unspecified) scann_->SetUnspecifiedParametersToDefaults(&p);
  }
  return params;
}

StatusOr<ScannConfig> ScannInterface::RetrainAndReindex(const string& config) {
  absl::Mutex mu;
  ScannConfig new_config = config_;
  if (!config.empty())
    SCANN_RETURN_IF_ERROR(ParseTextProto(&new_config, config));

  auto status_or = RetrainAndReindexSearcher(scann_.get(), &mu, new_config,
                                             parallel_query_pool_);
  if (!status_or.ok()) return status_or.status();
  scann_.reset(static_cast<SingleMachineSearcherBase<float>*>(
      std::move(status_or.value().release())));
  if (scann_->config().has_value()) config_ = scann_->config().value();
  scann_->MaybeReleaseDataset();
  return config_;
}

Status ScannInterface::Search(const DatapointPtr<float> query,
                              NNResultsVector* res, int final_nn,
                              int pre_reorder_nn, int leaves) const {
  if (query.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  SearchParameters params =
      GetSearchParameters(final_nn, pre_reorder_nn, leaves);
  scann_->SetUnspecifiedParametersToDefaults(&params);
  return scann_->FindNeighbors(query, params, res);
}

Status ScannInterface::SearchBatched(const DenseDataset<float>& queries,
                                     MutableSpan<NNResultsVector> res,
                                     int final_nn, int pre_reorder_nn,
                                     int leaves) const {
  if (queries.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  if (!std::isinf(scann_->default_pre_reordering_epsilon()) ||
      !std::isinf(scann_->default_post_reordering_epsilon()))
    return InvalidArgumentError("Batch querying isn't supported with epsilon");
  auto params = GetSearchParametersBatched(queries.size(), final_nn,
                                           pre_reorder_nn, leaves, true);
  return scann_->FindNeighborsBatched(queries, params, MakeMutableSpan(res));
}

Status ScannInterface::SearchBatchedParallel(const DenseDataset<float>& queries,
                                             MutableSpan<NNResultsVector> res,
                                             int final_nn, int pre_reorder_nn,
                                             int leaves, int batch_size) const {
  SCANN_RET_CHECK_EQ(queries.dimensionality(), dimensionality_);
  const size_t numQueries = queries.size();
  const size_t numCPUs = parallel_query_pool_->NumThreads();

  const size_t kBatchSize =
      std::min(std::max(min_batch_size_, DivRoundUp(numQueries, numCPUs)),
               static_cast<size_t>(batch_size));
  return ParallelForWithStatus<1>(
      Seq(DivRoundUp(numQueries, kBatchSize)), parallel_query_pool_.get(),
      [&](size_t i) {
        size_t begin = kBatchSize * i;
        size_t curSize = std::min(numQueries - begin, kBatchSize);
        vector<float> queryCopy(
            queries.data().begin() + begin * dimensionality_,
            queries.data().begin() + (begin + curSize) * dimensionality_);
        DenseDataset<float> curQueryDataset(std::move(queryCopy), curSize);
        return SearchBatched(curQueryDataset, res.subspan(begin, curSize),
                             final_nn, pre_reorder_nn, leaves);
      });
}

StatusOr<ScannAssets> ScannInterface::Serialize(std::string path,
                                                bool relative_path) {
  SCANN_ASSIGN_OR_RETURN(auto opts,
                         scann_->ExtractSingleMachineFactoryOptions());
  ScannAssets assets;
  const auto add_asset = [&assets](const std::string& fpath,
                                   ScannAsset::AssetType type) {
    ScannAsset* asset = assets.add_assets();
    asset->set_asset_type(type);
    asset->set_asset_path(fpath);
  };

  const auto convert_path = [&path, &relative_path](const std::string& fpath) {
    std::string absolute_path = path + "/" + fpath;
    return std::pair(relative_path ? fpath : absolute_path, absolute_path);
  };

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", config_));
  if (opts.ah_codebook != nullptr) {
    auto [rpath, fpath] = convert_path("ah_codebook.pb");
    add_asset(rpath, ScannAsset::AH_CENTERS);
    SCANN_RETURN_IF_ERROR(WriteProtobufToFile(fpath, *opts.ah_codebook));
  }
  if (opts.serialized_partitioner != nullptr) {
    auto [rpath, fpath] = convert_path("serialized_partitioner.pb");
    add_asset(rpath, ScannAsset::PARTITIONER);
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(fpath, *opts.serialized_partitioner));
  }
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token;
    if (HasSoar(config_)) {
      datapoint_to_token = vector<int32_t>(2 * n_points(), kSoarEmptyToken);
      for (const auto& [token_idx, dps] :
           Enumerate(*opts.datapoints_by_token)) {
        for (auto dp_idx : dps) {
          dp_idx *= 2;
          if (datapoint_to_token[dp_idx] != -1) dp_idx++;
          DCHECK_EQ(datapoint_to_token[dp_idx], -1);
          datapoint_to_token[dp_idx] = token_idx;
        }
      }
    } else {
      datapoint_to_token = vector<int32_t>(n_points());
      for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
        for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    }
    auto [rpath, fpath] = convert_path("datapoint_to_token.npy");
    add_asset(rpath, ScannAsset::TOKENIZATION_NPY);
    SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    auto [rpath, fpath] = convert_path("hashed_dataset.npy");
    add_asset(rpath, ScannAsset::AH_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.hashed_dataset)));

    if (opts.soar_hashed_dataset != nullptr) {
      DCHECK(HasSoar(config_));
      auto [rpath, fpath] = convert_path("hashed_dataset_soar.npy");
      add_asset(rpath, ScannAsset::AH_DATASET_SOAR_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.soar_hashed_dataset)));
    }
  }
  if (opts.bfloat16_dataset != nullptr) {
    auto [rpath, fpath] = convert_path("bfloat16_dataset.npy");
    add_asset(rpath, ScannAsset::BF16_DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *(opts.bfloat16_dataset)));
  }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      auto [rpath, fpath] = convert_path("int8_dataset.npy");
      add_asset(rpath, ScannAsset::INT8_DATASET_NPY);
      SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      auto [rpath, fpath] = convert_path("int8_multipliers.npy");
      add_asset(rpath, ScannAsset::INT8_MULTIPLIERS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      auto [rpath, fpath] = convert_path("dp_norms.npy");
      add_asset(rpath, ScannAsset::INT8_NORMS_NPY);
      SCANN_RETURN_IF_ERROR(VectorToNumpy(fpath, *norms));
    }
  }
  SCANN_ASSIGN_OR_RETURN(auto dataset, Float32DatasetIfNeeded());
  if (dataset != nullptr) {
    auto [rpath, fpath] = convert_path("dataset.npy");
    add_asset(rpath, ScannAsset::DATASET_NPY);
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(fpath, *dataset));
  }
  return assets;
}

StatusOr<ScannInterface::ScannHealthStats> ScannInterface::GetHealthStats()
    const {
  return scann_->GetHealthStats();
}

Status ScannInterface::InitializeHealthStats() {
  return scann_->InitializeHealthStats();
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

}  // namespace research_scann
