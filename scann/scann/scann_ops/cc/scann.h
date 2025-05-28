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

#ifndef SCANN_SCANN_OPS_CC_SCANN_H_
#define SCANN_SCANN_OPS_CC_SCANN_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/scann_assets.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/threads.h"

namespace research_scann {

class ScannInterface {
 public:
  using ScannArtifacts =
      std::tuple<ScannConfig, shared_ptr<DenseDataset<float>>,
                 SingleMachineFactoryOptions>;

  static StatusOr<ScannArtifacts> LoadArtifacts(const ScannConfig& config,
                                                const ScannAssets& orig_assets);
  static StatusOr<ScannArtifacts> LoadArtifacts(
      const std::string& artifacts_dir,
      const std::string& scann_assets_pbtxt = "");

  static StatusOr<std::unique_ptr<SingleMachineSearcherBase<float>>>
  CreateSearcher(ScannArtifacts artifacts);

  Status Initialize(const std::string& config_pbtxt,
                    const std::string& scann_assets_pbtxt);
  Status Initialize(ScannConfig config, SingleMachineFactoryOptions opts,
                    ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points);
  Status Initialize(ConstSpan<float> dataset, DatapointIndex n_points,
                    const std::string& config, int training_threads);
  Status Initialize(ScannArtifacts artifacts);

  StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
      const {
    return scann_->GetMutator();
  }

  StatusOr<ScannConfig> RetrainAndReindex(const string& config);

  Status Search(const DatapointPtr<float> query, NNResultsVector* res,
                int final_nn, int pre_reorder_nn, int leaves) const;
  Status SearchBatched(const DenseDataset<float>& queries,
                       MutableSpan<NNResultsVector> res, int final_nn,
                       int pre_reorder_nn, int leaves) const;
  Status SearchBatchedParallel(const DenseDataset<float>& queries,
                               MutableSpan<NNResultsVector> res, int final_nn,
                               int pre_reorder_nn, int leaves,
                               int batch_size = 256) const;
  StatusOr<ScannAssets> Serialize(std::string path, bool relative_path = false);
  StatusOr<SingleMachineFactoryOptions> ExtractOptions();

  template <typename T_idx>
  void ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                       float* distances);
  template <typename T_idx>
  void ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res, T_idx* indices,
                              float* distances, int neighbors_per_query);

  StatusOr<shared_ptr<const DenseDataset<float>>> Float32DatasetIfNeeded() {
    return scann_->SharedFloatDatasetIfNeeded();
  }

  size_t n_points() const { return scann_->DatasetSize().value(); }
  DimensionIndex dimensionality() const { return dimensionality_; }
  const ScannConfig* config() {
    if (scann_->config().has_value()) config_ = *scann_->config();
    return &config_;
  }

  std::shared_ptr<ThreadPool> parallel_query_pool() const {
    return parallel_query_pool_;
  }

  void SetNumThreads(int num_threads) {
    parallel_query_pool_ = StartThreadPool("ScannQueryingPool", num_threads);
  }

  using ScannHealthStats = SingleMachineSearcherBase<float>::HealthStats;
  StatusOr<ScannHealthStats> GetHealthStats() const;
  Status InitializeHealthStats();

 private:
  SearchParameters GetSearchParameters(int final_nn, int pre_reorder_nn,
                                       int leaves) const;
  vector<SearchParameters> GetSearchParametersBatched(
      int batch_size, int final_nn, int pre_reorder_nn, int leaves,
      bool set_unspecified) const;
  DimensionIndex dimensionality_;
  std::unique_ptr<SingleMachineSearcherBase<float>> scann_;
  ScannConfig config_;

  float result_multiplier_;

  size_t min_batch_size_;

  std::shared_ptr<ThreadPool> parallel_query_pool_;
};

template <typename T_idx>
void ScannInterface::ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                                     float* distances) {
  for (const auto& p : res) {
    *(indices++) = static_cast<T_idx>(p.first);
    *(distances++) = result_multiplier_ * p.second;
  }
}

template <typename T_idx>
void ScannInterface::ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res,
                                            T_idx* indices, float* distances,
                                            int neighbors_per_query) {
  for (const auto& result_vec : res) {
    DCHECK_LE(result_vec.size(), neighbors_per_query);
    for (const auto& pair : result_vec) {
      *(indices++) = static_cast<T_idx>(pair.first);
      *(distances++) = result_multiplier_ * pair.second;
    }

    for (int i = result_vec.size(); i < neighbors_per_query; i++) {
      *(indices++) = 0;
      *(distances++) = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

template <typename T>
Status ParseTextProto(T* proto, const string& proto_str) {
  ::google::protobuf::TextFormat::ParseFromString(proto_str, proto);
  return OkStatus();
}

}  // namespace research_scann

#endif
