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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
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
#include "scann/utils/threads.h"

namespace tensorflow {
namespace scann_ops {

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  ConstSpan<int32_t> datapoint_to_token,
                                  ConstSpan<uint8_t> hashed_dataset,
                                  DimensionIndex dimensionality,
                                  const std::string& artifacts_dir) {
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
                    dimensionality);
}

Status ScannInterface::Initialize(ScannConfig config,
                                  SingleMachineFactoryOptions opts,
                                  ConstSpan<float> dataset,
                                  ConstSpan<int32_t> datapoint_to_token,
                                  ConstSpan<uint8_t> hashed_dataset,
                                  DimensionIndex dimensionality) {
  config_ = config;
  if (opts.ah_codebook != nullptr) {
    vector<uint8_t> hashed_db(hashed_dataset.data(),
                              hashed_dataset.data() + hashed_dataset.size());
    int n_points = dataset.size() / dimensionality;
    opts.hashed_dataset =
        std::make_shared<DenseDataset<uint8_t>>(hashed_db, n_points);
  }
  if (opts.serialized_partitioner != nullptr) {
    if (datapoint_to_token.size() * dimensionality != dataset.size())
      return InvalidArgumentError(
          "Sizes of datapoint_to_token and dataset are inconsistent");
    opts.datapoints_by_token =
        std::make_shared<vector<std::vector<DatapointIndex>>>(
            opts.serialized_partitioner->n_tokens());
    auto ptr = datapoint_to_token.data();
    for (int i = 0; i < datapoint_to_token.size(); i++) {
      opts.datapoints_by_token->at(ptr[i]).push_back(i);
    }
  }
  return Initialize(dataset, dimensionality, opts);
}

Status ScannInterface::Initialize(ConstSpan<float> dataset,
                                  DimensionIndex dimensionality,
                                  const std::string& config,
                                  int training_threads) {
  ::google::protobuf::TextFormat::ParseFromString(config, &config_);
  if (training_threads < 0)
    return InvalidArgumentError("training_threads must be non-negative");
  if (training_threads == 0) training_threads = absl::base_internal::NumCPUs();
  SingleMachineFactoryOptions opts;

  opts.parallelization_pool =
      StartThreadPool("scann_threadpool", training_threads - 1);
  return Initialize(dataset, dimensionality, opts);
}

Status ScannInterface::Initialize(ConstSpan<float> ds_span,
                                  DimensionIndex dimensionality,
                                  SingleMachineFactoryOptions opts) {
  if (ds_span.empty()) return InvalidArgumentError("Dataset must be non-empty");

  dimensionality_ = dimensionality;
  n_points_ = ds_span.size() / dimensionality_;

  vector<float> dataset_vec(ds_span.data(), ds_span.data() + ds_span.size());
  auto dataset = absl::make_unique<DenseDataset<float>>(dataset_vec, n_points_);

  if (config_.has_partitioning() &&
      config_.partitioning().partitioning_type() ==
          PartitioningConfig::SPHERICAL)
    dataset->set_normalization_tag(tensorflow::scann_ops::UNITL2NORM);
  TF_ASSIGN_OR_RETURN(scann_,
                      SingleMachineFactory<float>(config_, std::move(dataset),
                                                  std::move(opts)));

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
    for (int i = 0; i < opts.datapoints_by_token->size(); i++) {
      for (auto dp_idx : opts.datapoints_by_token->at(i))
        datapoint_to_token[dp_idx] = i;
    }
    SCANN_RETURN_IF_ERROR(
        VectorToNumpy(path + "/datapoint_to_token.npy", datapoint_to_token));
  }
  if (opts.hashed_dataset != nullptr) {
    SCANN_RETURN_IF_ERROR(
        DatasetToNumpy(path + "/hashed_dataset.npy", *(opts.hashed_dataset)));
  }
  return OkStatus();
}

StatusOr<SingleMachineFactoryOptions> ScannInterface::ExtractOptions() {
  return scann_->ExtractSingleMachineFactoryOptions();
}

}  // namespace scann_ops
}  // namespace tensorflow
