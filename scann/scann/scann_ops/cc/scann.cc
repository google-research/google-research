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

#include "scann/scann_ops/cc/scann.h"

#include <fstream>

#include "absl/base/internal/sysinfo.h"
#include "absl/container/node_hash_set.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/io_npy.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/threads.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
void ParseTextProto(T* proto, const std::string& proto_str) {
  ::google::protobuf::TextFormat::ParseFromString(proto_str, proto);
}

unique_ptr<DenseDataset<float>> InitDataset(ConstSpan<float> dataset,
                                            DatapointIndex n_points) {
  if (dataset.empty()) return nullptr;

  vector<float> dataset_vec(dataset.data(), dataset.data() + dataset.size());
  return absl::make_unique<DenseDataset<float>>(dataset_vec, n_points);
}

Status ScannInterface::Initialize(
    ConstSpan<float> dataset, ConstSpan<int32_t> datapoint_to_token,
    ConstSpan<uint8_t> hashed_dataset, ConstSpan<int8_t> int8_dataset,
    ConstSpan<float> int8_multipliers, ConstSpan<float> dp_norms,
    DatapointIndex n_points, const std::string& artifacts_dir) {
  ScannConfig config;
  SCANN_RETURN_IF_ERROR(
      ReadProtobufFromFile(artifacts_dir + "/scann_config.pb", &config));
  SingleMachineFactoryOptions opts;
  if (!hashed_dataset.empty()) {
    opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
    SCANN_RETURN_IF_ERROR(ReadProtobufFromFile(
        artifacts_dir + "/ah_codebook.pb", opts.ah_codebook.get()));
  }
  if (!datapoint_to_token.empty()) {
    opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
    SCANN_RETURN_IF_ERROR(
        ReadProtobufFromFile(artifacts_dir + "/serialized_partitioner.pb",
                             opts.serialized_partitioner.get()));
  }
  return Initialize(config, opts, dataset, datapoint_to_token, hashed_dataset,
                    int8_dataset, int8_multipliers, dp_norms, n_points);
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
        std::make_shared<DenseDataset<uint8_t>>(hashed_db, n_points);
  }
  if (opts.serialized_partitioner != nullptr) {
    if (datapoint_to_token.size() != n_points)
      return InvalidArgumentError(
          absl::StrFormat("datapoint_to_token length=%d but expected %d",
                          datapoint_to_token.size(), n_points));
    opts.datapoints_by_token =
        std::make_shared<vector<std::vector<DatapointIndex>>>(
            opts.serialized_partitioner->n_tokens());
    for (auto [dp_idx, token] : Enumerate(datapoint_to_token))
      opts.datapoints_by_token->at(token).push_back(dp_idx);
  }
  if (!int8_dataset.empty()) {
    auto int8_data = std::make_shared<PreQuantizedFixedPoint>();
    vector<int8_t> int8_vec(int8_dataset.data(),
                            int8_dataset.data() + int8_dataset.size());
    int8_data->fixed_point_dataset =
        std::make_shared<DenseDataset<int8_t>>(int8_vec, n_points);

    int8_data->multiplier_by_dimension = make_shared<vector<float>>(
        int8_multipliers.begin(), int8_multipliers.end());

    int8_data->squared_l2_norm_by_datapoint =
        make_shared<vector<float>>(dp_norms.begin(), dp_norms.end());
    opts.pre_quantized_fixed_point = int8_data;
  }
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DatapointIndex n_points,
                                  const std::string& config,
                                  int training_threads) {
  ParseTextProto(&config_, config);
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = absl::base_internal::NumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);
  return Initialize(InitDataset(dataset, n_points), opts);
}

Status ScannInterface::Initialize(shared_ptr<DenseDataset<float>> dataset,
                                  SingleMachineFactoryOptions opts) {
  TF_ASSIGN_OR_RETURN(dimensionality_, opts.ComputeConsistentDimensionality(
                                           config_.hash(), dataset.get()));
  TF_ASSIGN_OR_RETURN(n_points_, opts.ComputeConsistentSize(dataset.get()));

  if (dataset && config_.has_partitioning() &&
      config_.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(tensorflow::scann_ops::UNITL2NORM);
  TF_ASSIGN_OR_RETURN(scann_, SingleMachineFactoryScann<float>(
                                  config_, dataset, std::move(opts)));

  const std::string& distance = config_.distance_measure().distance_measure();
  const absl::node_hash_set<std::string> negated_distances{
      "DotProductDistance", "BinaryDotProductDistance", "AbsDotProductDistance",
      "LimitedInnerProductDistance"};
  result_multiplier_ =
      negated_distances.find(distance) == negated_distances.end() ? 1 : -1;
  return OkStatus();
}

Status ScannInterface::Search(const DatapointPtr<float> query,
                              NNResultsVector* res, int final_nn,
                              int pre_reorder_nn, int leaves) const {
  if (query.dimensionality() != dimensionality_)
    return InvalidArgumentError("Query doesn't match dataset dimsensionality");
  bool has_reordering =
      config_.has_exact_reordering() || config_.has_compressed_reordering();
  int post_reorder_nn = -1;
  if (has_reordering)
    post_reorder_nn = final_nn;
  else
    pre_reorder_nn = final_nn;

  SearchParameters params;
  params.set_pre_reordering_num_neighbors(pre_reorder_nn);
  params.set_post_reordering_num_neighbors(post_reorder_nn);
  if (leaves > 0) {
    auto tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
    params.set_searcher_specific_optional_parameters(tree_params);
  }
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
  bool has_reordering =
      config_.has_exact_reordering() || config_.has_compressed_reordering();
  int post_reorder_nn = -1;
  if (has_reordering)
    post_reorder_nn = final_nn;
  else
    pre_reorder_nn = final_nn;

  std::vector<SearchParameters> params(queries.size());
  std::shared_ptr<tensorflow::scann_ops::TreeXOptionalParameters> tree_params;
  if (leaves > 0) {
    tree_params = std::make_shared<TreeXOptionalParameters>();
    tree_params->set_num_partitions_to_search_override(leaves);
  }

  for (auto& p : params) {
    p.set_pre_reordering_num_neighbors(pre_reorder_nn);
    p.set_post_reordering_num_neighbors(post_reorder_nn);
    if (tree_params) p.set_searcher_specific_optional_parameters(tree_params);
    scann_->SetUnspecifiedParametersToDefaults(&p);
  }

  return scann_->FindNeighborsBatched(queries, params, MakeMutableSpan(res));
}

Status ScannInterface::SearchBatchedParallel(const DenseDataset<float>& queries,
                                             MutableSpan<NNResultsVector> res,
                                             int final_nn, int pre_reorder_nn,
                                             int leaves) const {
  const size_t numQueries = queries.size();
  const size_t kBatchSize = 256;
  auto pool = StartThreadPool("pool", absl::base_internal::NumCPUs() - 1);
  return ParallelForWithStatus<1>(
      Seq(DivRoundUp(numQueries, kBatchSize)), pool.get(), [&](size_t i) {
        size_t begin = kBatchSize * i;
        size_t curSize = std::min(numQueries - begin, kBatchSize);
        vector<float> queryCopy(
            queries.data().begin() + begin * dimensionality_,
            queries.data().begin() + (begin + curSize) * dimensionality_);
        DenseDataset<float> curQueryDataset(queryCopy, curSize);
        return SearchBatched(curQueryDataset, {res.begin() + begin, curSize},
                             final_nn, pre_reorder_nn, leaves);
      });
}

Status ScannInterface::Serialize(std::string path) {
  TF_ASSIGN_OR_RETURN(auto opts, scann_->ExtractSingleMachineFactoryOptions());

  SCANN_RETURN_IF_ERROR(
      WriteProtobufToFile(path + "/scann_config.pb", &config_));
  if (opts.ah_codebook != nullptr)
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(path + "/ah_codebook.pb", opts.ah_codebook.get()));
  if (opts.serialized_partitioner != nullptr)
    SCANN_RETURN_IF_ERROR(
        WriteProtobufToFile(path + "/serialized_partitioner.pb",
                            opts.serialized_partitioner.get()));
  if (opts.datapoints_by_token != nullptr) {
    vector<int32_t> datapoint_to_token(n_points_);
    for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
      for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(path + "/datapoint_to_token.npy", datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    SCANN_RETURN_IF_ERROR(
        DatasetToNumpy(path + "/hashed_dataset.npy", *(opts.hashed_dataset)));
  }
  if (opts.pre_quantized_fixed_point != nullptr) {
    auto fixed_point = opts.pre_quantized_fixed_point;
    auto dataset = fixed_point->fixed_point_dataset;
    if (dataset != nullptr) {
      SCANN_RETURN_IF_ERROR(
          DatasetToNumpy(path + "/int8_dataset.npy", *dataset));
    }
    auto multipliers = fixed_point->multiplier_by_dimension;
    if (multipliers != nullptr) {
      SCANN_RETURN_IF_ERROR(
          VectorToNumpy(path + "/int8_multipliers.npy", *multipliers));
    }
    auto norms = fixed_point->squared_l2_norm_by_datapoint;
    if (norms != nullptr) {
      SCANN_RETURN_IF_ERROR(VectorToNumpy(path + "/dp_norms.npy", *norms));
    }
  }
  if (scann_->needs_dataset()) {
    if (scann_->dataset() == nullptr)
      return InternalError(
          "Searcher needs original dataset but none is present.");
    auto dataset = dynamic_cast<const DenseDataset<float>*>(scann_->dataset());
    if (dataset == nullptr)
      return InternalError("Failed to cast dataset to DenseDataset<float>.");
    SCANN_RETURN_IF_ERROR(DatasetToNumpy(path + "/dataset.npy", *dataset));
  }
  return OkStatus();
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

}  // namespace scann_ops
}  // namespace tensorflow
