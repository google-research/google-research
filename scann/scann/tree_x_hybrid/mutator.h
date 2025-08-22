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



#ifndef SCANN_TREE_X_HYBRID_MUTATOR_H_
#define SCANN_TREE_X_HYBRID_MUTATOR_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "scann/base/health_stats_collector.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

class TreeAHHybridResidual;
class TreeXHybridIncrementalOptions;

template <typename Searcher>
class TreeXHybridMutator
    : public SingleMachineSearcherBase<typename Searcher::DataType>::Mutator {
 public:
  using T = typename Searcher::DataType;
  using PrecomputedMutationArtifacts =
      UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
  using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
  using MutateBaseOptions =
      UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

  using SingleMachineSearcherBase<
      typename Searcher::DataType>::Mutator::AddDatapoint;
  using SingleMachineSearcherBase<
      typename Searcher::DataType>::Mutator::UpdateDatapoint;

  struct TreeXPrecomputedMutationArtifacts final
      : public PrecomputedMutationArtifacts {
    TreeXPrecomputedMutationArtifacts() = default;

    DatapointPtr<T> GetMaybeResidual(const DatapointPtr<T>& dptr,
                                     size_t token_idx) const {
      if (!kIsTreeAHResidual) return dptr;
      return MakeDatapointPtr<T>(
          residual_storage[0].get() + token_idx * residual_dimensionality[0],
          residual_dimensionality[0]);
    }

    vector<int32_t> tokens;
    vector<unique_ptr<PrecomputedMutationArtifacts>> leaf_precomputed_artifacts;

    enum { kIsTreeAHResidual = std::is_same_v<Searcher, TreeAHHybridResidual> };
    unique_ptr<T[]> residual_storage[kIsTreeAHResidual];

    uint32_t residual_dimensionality[kIsTreeAHResidual];
    static_assert(kIsTreeAHResidual || sizeof(residual_storage) == 0);
    static_assert(kIsTreeAHResidual || sizeof(residual_dimensionality) == 0);
    SCANN_DECLARE_MOVE_ONLY_CLASS(TreeXPrecomputedMutationArtifacts);
  };

  static StatusOr<unique_ptr<TreeXHybridMutator<Searcher>>> Create(
      Searcher* searcher);
  TreeXHybridMutator(const TreeXHybridMutator&) = delete;
  TreeXHybridMutator& operator=(const TreeXHybridMutator&) = delete;
  ~TreeXHybridMutator() final {}
  unique_ptr<PrecomputedMutationArtifacts> ComputePrecomputedMutationArtifacts(
      const DatapointPtr<T>& dptr) const final;
  vector<unique_ptr<PrecomputedMutationArtifacts>>

  ComputePrecomputedMutationArtifacts(const TypedDataset<T>& batch) const final;

  absl::StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex i) const override;
  DatapointPtr<T> GetDatapointPtr(DatapointIndex i, Datapoint<T>* storage,
                                  bool always_copy = false) const;

  vector<unique_ptr<PrecomputedMutationArtifacts>>
  ComputePrecomputedMutationArtifacts(
      const TypedDataset<T>& ds,
      std::shared_ptr<ThreadPool> thread_pool) const final;

  Status AddCentroid(DatapointPtr<float> x);
  Status RemoveCentroid(int32_t token);
  Status UpdateCentroid(DatapointPtr<float> x, int32_t token);

  Status EnableIncrementalTraining(const ScannConfig& config) final;
  Status EnableIncrementalTraining(shared_ptr<TreeXHybridIncrementalOptions>);

  void CheckReassignment(int32_t token,
                         const MutationOptions& mo = MutationOptions());

  Status ShrinkPartition(int32_t token = kInvalidToken);

  Status SplitPartition(int32_t token = kInvalidToken);

  Status IngestUpdate(DatapointIndex token, DatapointPtr<T> x, int delta);

  Status Reassign(ConstSpan<int32_t> tokens, const MutationOptions& mo);
  Status Reassign(int32_t token, const MutationOptions& mo) {
    return Reassign({&token, 1}, mo);
  }

  Status ReassignTopK(int32_t token, int32_t max_split,
                      const MutationOptions& mo);

  StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                        string_view docid,
                                        const MutationOptions& mo) final;
  Status RemoveDatapoint(string_view docid) final;
  void Reserve(size_t size) final;
  StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                           string_view docid,
                                           const MutationOptions& mo) final;

  StatusOr<std::optional<ScannConfig>> IncrementalMaintenance() final;

  Status RemoveDatapoint(DatapointIndex index) final;
  StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                           DatapointIndex index,
                                           const MutationOptions& mo) final;

  unique_ptr<PrecomputedMutationArtifacts>
  ComputeLeafPrecomputedMutationArtifacts(
      int32_t token, const DatapointPtr<T>& maybe_residual,
      const DatapointPtr<T>& original) const;

  Status CheckGlobalToLocalConsistency() const;

  shared_ptr<TreeXHybridIncrementalOptions> incremental_opts_ = nullptr;

 private:
  using GlobalToLocal1 = vector<std::array<pair<int, DatapointIndex>, 1>>;
  using GlobalToLocal2 = vector<std::array<pair<int, DatapointIndex>, 2>>;

  TreeXHybridMutator(
      Searcher* searcher,
      vector<typename SingleMachineSearcherBase<T>::Mutator*> leaf_mutators,
      std::variant<GlobalToLocal1, GlobalToLocal2> global_to_local)
      : searcher_(searcher),
        leaf_mutators_(std::move(leaf_mutators)),
        global_to_local_(std::move(global_to_local)) {}

  template <typename GlobalToLocal>
  StatusOr<DatapointIndex> AddDatapoint(
      const DatapointPtr<T>& dptr, string_view docid,
      const TreeXPrecomputedMutationArtifacts& ma, const MutationOptions& mo);

  template <typename GlobalToLocal>
  StatusOr<DatapointIndex> UpdateDatapoint(
      const DatapointPtr<T>& dptr, DatapointIndex dp_idx,
      const TreeXPrecomputedMutationArtifacts& ma, const MutationOptions& mo);

  template <typename GlobalToLocal>
  Status RemoveDatapointImpl(DatapointIndex dp_idx);

  template <typename GlobalToLocal>
  Status UpdateSubIndex(DatapointIndex dp_idx, int token_idx,
                        DatapointIndex new_sub_idx) {
    auto& global_to_local = std::get<GlobalToLocal>(global_to_local_);
    SCANN_RET_CHECK_NE(token_idx, kInvalidToken);
    if (dp_idx >= global_to_local.size()) {
      Status status = NotFoundError(
          absl::StrFormat("Cannot update subindex for non-existent datapoint "
                          "idx %d (token_idx = %d)",
                          dp_idx, token_idx));
      return status;
    }

    for (auto& pair : global_to_local[dp_idx]) {
      if (pair.first == token_idx) {
        pair.second = new_sub_idx;
        return OkStatus();
      }
    }

    Status status = NotFoundError(
        absl::StrFormat("Cannot update subindex for non-existent token "
                        "idx %d (dp_idx = %d)",
                        token_idx, dp_idx));
    return status;
  }

  template <typename ArrayT>
  static void InitializeGlobalToLocalEntry(ArrayT& x) {
    std::fill(x.begin(), x.end(),
              std::make_pair(kInvalidToken, kInvalidDatapointIndex));
  }

  Status InitializeCentroids();

  template <typename GlobalToLocalT>
  static GlobalToLocalT CreateGlobalToLocal(
      MutableSpan<std::vector<DatapointIndex>> datapoints_by_token,
      size_t num_datapoints) {
    if constexpr (std::is_same_v<GlobalToLocalT, GlobalToLocal1>) {
      GlobalToLocal1 result(num_datapoints);
      for (size_t i : IndicesOf(datapoints_by_token)) {
        for (size_t j : IndicesOf(datapoints_by_token[i])) {
          result[datapoints_by_token[i][j]][0] = std::make_pair(i, j);
        }
      }
      return result;
    } else {
      vector<std::vector<pair<int, DatapointIndex>>> tmp(num_datapoints);
      for (size_t i : IndicesOf(datapoints_by_token)) {
        for (size_t j : IndicesOf(datapoints_by_token[i])) {
          tmp[datapoints_by_token[i][j]].push_back(std::make_pair(i, j));
        }
      }

      GlobalToLocal2 global_to_local(num_datapoints);
      for (size_t i : IndicesOf(tmp)) {
        InitializeGlobalToLocalEntry(global_to_local[i]);
        for (size_t j : IndicesOf(tmp[i])) {
          global_to_local[i][j] = tmp[i][j];
        }
      }
      return global_to_local;
    }
  }

  Status UpgradeGlobalToLocalIfNeeded(size_t num_tokens) {
    if (num_tokens <= 1) return OkStatus();
    SCANN_RET_CHECK_LE(num_tokens, 2)
        << "Spilling to >2 centroids isn't supported in tree-X.  This is "
           "enforced at several levels of abstraction.  This shouldn't be "
           "possible.";
    if (!is_disjoint()) return OkStatus();

    global_to_local_ = GlobalToLocal2();
    global_to_local_ = CreateGlobalToLocal<GlobalToLocal2>(
        MakeMutableSpan(searcher_->datapoints_by_token_),
        searcher_->num_datapoints_);
    return OkStatus();
  }

  bool is_disjoint() const {
    return std::holds_alternative<GlobalToLocal1>(global_to_local_);
  }

  Status UpdateCentroid(DatapointPtr<float> x, int32_t token,
                        bool stats_update);

  Status RemoveEmptyCentroid(int32_t token);

  using HealthStatsCollector = typename Searcher::HealthStatsCollector;
  HealthStatsCollector& stats_collector() const {
    return searcher_->stats_collector_;
  }

  Searcher* searcher_ = nullptr;

  enum : int { kInvalidToken = -1 };

  vector<typename SingleMachineSearcherBase<T>::Mutator*> leaf_mutators_;

  std::variant<GlobalToLocal1, GlobalToLocal2> global_to_local_;

  std::vector<float> mutation_stats_;

  std::shared_ptr<KMeansTreePartitioner<typename Searcher::DataType>> centroid_;

  absl::flat_hash_set<int32_t> reassignment_set_;
  std::vector<Datapoint<float>> old_centroids_;

  int32_t updated_token_centroid_ = kInvalidToken;
};

class TreeXHybridIncrementalOptions {
 public:
  enum : int { kInvalidToken = -1 };
  std::variant<float, DatapointIndex> incremental_threshold_;
  int32_t cluster_stability_size_ = 200;
  uint32_t max_split_ = std::numeric_limits<uint32_t>::max();
  bool enable_health_stats_collection_ = false;
};

#define SCANN_DISPATCH_ON_GLOBAL_TO_LOCAL(function, ...)            \
  [&]() {                                                           \
    if (std::holds_alternative<GlobalToLocal1>(global_to_local_)) { \
      return function<GlobalToLocal1>(__VA_ARGS__);                 \
    } else {                                                        \
      return function<GlobalToLocal2>(__VA_ARGS__);                 \
    }                                                               \
  }();

template <typename T>
std::vector<float> gradient_update(DatapointPtr<float> w, DatapointPtr<T> x,
                                   const float lr) {
  std::vector<float> res;
  res.reserve(w.dimensionality());
  for (DatapointIndex i = 0; i < w.dimensionality(); ++i) {
    res.push_back(w.values_span()[i] * (1 - lr) + x.values_span()[i] * lr);
  }
  return res;
}

template <typename Searcher>
absl::StatusOr<Datapoint<typename Searcher::DataType>>
TreeXHybridMutator<Searcher>::GetDatapoint(DatapointIndex i) const {
  size_t size = 0;
  if (searcher_->shared_dataset()) {
    size = searcher_->shared_dataset()->size();
  } else if (searcher_->reordering_enabled()) {
    size = searcher_->reordering_helper().dataset()->size();
  } else {
    if (std::holds_alternative<GlobalToLocal1>(global_to_local_)) {
      auto& global_to_local = std::get<GlobalToLocal1>(global_to_local_);
      size = global_to_local.size();
    } else {
      auto& global_to_local = std::get<GlobalToLocal2>(global_to_local_);
      size = global_to_local.size();
    }
  }
  if (i >= size) {
    return OutOfRangeError(
        "Datapoint index out of bound: index = %d, but size = %d.", i, size);
  }

  Datapoint<T> dp;
  GetDatapointPtr(i, &dp, true);
  return dp;
}

template <typename Searcher>
DatapointPtr<typename Searcher::DataType>
TreeXHybridMutator<Searcher>::GetDatapointPtr(
    DatapointIndex i, Datapoint<typename Searcher::DataType>* storage,
    bool always_copy) const {
  DatapointPtr<T> x;
  if (searcher_->shared_dataset()) {
    x = searcher_->GetDatapointPtr(i);
  } else if (searcher_->reordering_enabled()) {
    auto dim = searcher_->reordering_helper().dataset()->dimensionality();
    Datapoint<float> dp;
    dp.mutable_values()->resize(dim);
    searcher_->reordering_helper()
        .Reconstruct(i, dp.mutable_values_span())
        .IgnoreError();
    CopyToDatapoint(dp.ToPtr(), storage);
    return storage->ToPtr();
  } else {
    if (std::holds_alternative<GlobalToLocal1>(global_to_local_)) {
      auto& global_to_local = std::get<GlobalToLocal1>(global_to_local_);
      auto idx = global_to_local[i][0];
      x = searcher_->leaf_searchers_[idx.first]->GetDatapointPtr(idx.second);
    } else {
      auto& global_to_local = std::get<GlobalToLocal2>(global_to_local_);
      auto idx = global_to_local[i][0];
      x = searcher_->leaf_searchers_[idx.first]->GetDatapointPtr(idx.second);
    }
  }
  if (always_copy) {
    CopyToDatapoint(x, storage);
    return storage->ToPtr();
  } else {
    return x;
  }
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::IngestUpdate(DatapointIndex token,
                                                  DatapointPtr<T> x,
                                                  int delta) {
  DatapointIndex n = searcher_->datapoints_by_token_[token].size() + delta;

  float lr = std::min(0.001, 1.0 / n) * delta;

  auto res = gradient_update(centroid_->LeafCenters()[token], x, lr);
  return UpdateCentroid(MakeDatapointPtr(res), token);
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::EnableIncrementalTraining(
    const ScannConfig& config) {
  if (config.partitioning().has_incremental_training_config()) {
    if (searcher_->dataset() == nullptr && !searcher_->reordering_enabled() &&
        searcher_->leaf_searchers_[0]->dataset() == nullptr) {
      return FailedPreconditionError(
          "Incremental training requires either the original float dataset or"
          " enabled with reordering.");
    }
    const auto& itc = config.partitioning().incremental_training_config();
    auto opts = make_shared<TreeXHybridIncrementalOptions>();
    if (itc.has_fraction())
      opts->incremental_threshold_ = itc.fraction();
    else
      opts->incremental_threshold_ = itc.number_of_datapoints();
    opts->cluster_stability_size_ = itc.cluster_stability_size();
    if (itc.max_split() <= 1)
      return FailedPreconditionError(absl::StrFormat(
          "max_split in incremental training must be larger than 1, got %d",
          itc.max_split()));
    opts->max_split_ = itc.max_split();
    SCANN_RETURN_IF_ERROR(this->EnableIncrementalTraining(opts));
    return OkStatus();
  } else {
    return FailedPreconditionError("Incremental training config not present.");
  }
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::EnableIncrementalTraining(
    shared_ptr<TreeXHybridIncrementalOptions> incremental_opts) {
  incremental_opts_ = incremental_opts;
  mutation_stats_.resize(searcher_->datapoints_by_token_.size(), 0.0f);
  if (incremental_opts_ && incremental_opts_->enable_health_stats_collection_) {
    SCANN_RETURN_IF_ERROR(searcher_->InitializeHealthStats());
  }

  return InitializeCentroids();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::InitializeCentroids() {
  auto pd = std::dynamic_pointer_cast<
      const KMeansTreePartitioner<typename Searcher::DataType>>(
      searcher_->database_tokenizer_);
  auto pq = std::dynamic_pointer_cast<
      const KMeansTreePartitioner<typename Searcher::DataType>>(
      searcher_->query_tokenizer_);
  SCANN_RET_CHECK(pq != nullptr)
      << "Query partitioner must be a KMeansTreeLikePartitioner.";
  SCANN_RET_CHECK_EQ(pd->kmeans_tree(), pq->kmeans_tree())
      << "Centroids in database partitioner and query partitioner must be "
      << "identical";
  SCANN_RET_CHECK(pq->kmeans_tree()->is_flat())
      << "The query/database partitioner must contain a single flat "
      << "KMeansTree.";

  centroid_ = std::const_pointer_cast<
      KMeansTreePartitioner<typename Searcher::DataType>>(pq);
  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::AddCentroid(DatapointPtr<float> x) {
  SCANN_RET_CHECK_NE(nullptr, centroid_)
      << "Incremental training must be enabled for AddCentroid.";

  SCANN_RETURN_IF_ERROR(searcher_->AddLeafSearcher());
  SCANN_ASSIGN_OR_RETURN(auto leaf_mutator,
                         searcher_->leaf_searchers_.back()->GetMutator());
  leaf_mutators_.push_back(leaf_mutator);
  mutation_stats_.push_back(0.0f);
  stats_collector().AddPartition();

  SCANN_ASSIGN_OR_RETURN(auto mutator, centroid_->LeafCenters().GetMutator());
  return mutator->AddDatapoint(x, "");
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::RemoveCentroid(int32_t token) {
  SCANN_RET_CHECK_LT(token, searcher_->datapoints_by_token_.size())
      << "Incorrect token number: " << token;
  auto& datapoints_by_token = searcher_->datapoints_by_token_;
  while (!datapoints_by_token[token].empty())
    SCANN_RETURN_IF_ERROR(RemoveDatapoint(datapoints_by_token[token].back()));
  return RemoveEmptyCentroid(token);
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::RemoveEmptyCentroid(int32_t token) {
  SCANN_RET_CHECK_NE(nullptr, centroid_)
      << "Incremental training must be enabled for RemoveCentroid.";
  SCANN_RET_CHECK_LT(token, searcher_->datapoints_by_token_.size())
      << "Incorrect token number: " << token;

  auto& datapoints_by_token = searcher_->datapoints_by_token_;
  SCANN_RET_CHECK(datapoints_by_token[token].empty());
  int32_t n_tokens = datapoints_by_token.size();

  if (token != n_tokens - 1) {
    datapoints_by_token[token] = std::move(datapoints_by_token[n_tokens - 1]);
    searcher_->leaf_searchers_[token] =
        std::move(searcher_->leaf_searchers_[n_tokens - 1]);
    leaf_mutators_[token] = std::move(leaf_mutators_[n_tokens - 1]);
    mutation_stats_[token] = mutation_stats_[n_tokens - 1];
    stats_collector().SwapPartitions(token, n_tokens - 1);

    if (std::holds_alternative<GlobalToLocal1>(global_to_local_)) {
      auto& global_to_local = std::get<GlobalToLocal1>(global_to_local_);
      for (DatapointIndex i : searcher_->datapoints_by_token_[token])
        for (pair<int, DatapointIndex>& p : global_to_local[i]) p.first = token;
    } else {
      auto& global_to_local = std::get<GlobalToLocal2>(global_to_local_);
      for (DatapointIndex i : searcher_->datapoints_by_token_[token])
        for (pair<int, DatapointIndex>& p : global_to_local[i]) p.first = token;
    }
  }

  SCANN_ASSIGN_OR_RETURN(auto mutator, centroid_->LeafCenters().GetMutator());
  searcher_->datapoints_by_token_.pop_back();
  searcher_->leaf_searchers_.pop_back();
  mutation_stats_.pop_back();
  leaf_mutators_.pop_back();
  stats_collector().RemoveLastPartition();
  return mutator->RemoveDatapoint(token);
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::ShrinkPartition(int32_t token) {
  auto& datapoints_by_token = searcher_->datapoints_by_token_;
  int32_t n_tokens = datapoints_by_token.size();

  if (token == kInvalidToken) {
    token = 0;
    for (int32_t i : Seq(n_tokens))
      if (datapoints_by_token[i].size() < datapoints_by_token[token].size())
        token = i;
  }
  VLOG(3) << "Shrink " << token << " / " << n_tokens
          << " Size = " << datapoints_by_token[token].size();

  stats_collector().SubtractPartition(token);
  if (incremental_opts_ && incremental_opts_->enable_health_stats_collection_)
    updated_token_centroid_ = token;

  vector<float> res(centroid_->LeafCenters()[0].dimensionality(),
                    numeric_limits<int32_t>::max());
  SCANN_RETURN_IF_ERROR(UpdateCentroid(MakeDatapointPtr(res), token, false));

  SCANN_RETURN_IF_ERROR(Reassign(token, {.reassignment_in_flight = true}));
  updated_token_centroid_ = kInvalidToken;

  if (reassignment_set_.contains(n_tokens - 1)) {
    reassignment_set_.erase(n_tokens - 1);
    reassignment_set_.insert(token);
  }

  return RemoveEmptyCentroid(token);
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::SplitPartition(int32_t token) {
  auto& datapoints_by_token = searcher_->datapoints_by_token_;

  int32_t n_tokens = datapoints_by_token.size();
  if (token == kInvalidToken) {
    token = 0;
    for (int32_t i : Seq(n_tokens))
      if (datapoints_by_token[i].size() > datapoints_by_token[token].size())
        token = i;
  }
  auto original_size = datapoints_by_token[token].size();

  if (original_size == 0)
    return InvalidArgumentError("Cannot split an empty partition.");

  Datapoint<float> centroid;
  Datapoint<T> storage;
  CopyToDatapoint(centroid_->LeafCenters()[token], &centroid);
  const float kEpsilon = 1e-6;
  DatapointPtr<T> perturb =
      GetDatapointPtr(datapoints_by_token[token][0], &storage, true);
  auto norm = std::sqrt(SquaredL2Norm(perturb)) + kEpsilon;
  for (int d : Seq(centroid.dimensionality()))
    centroid.mutable_values_span()[d] +=
        kEpsilon * perturb.values_span()[d] / norm;
  SCANN_RETURN_IF_ERROR(AddCentroid(centroid.ToPtr()));

  if (incremental_opts_->max_split_ >= n_tokens) {
    SCANN_RETURN_IF_ERROR(Reassign(token, {.reassignment_in_flight = true}));
  } else {
    SCANN_RETURN_IF_ERROR(ReassignTopK(token, incremental_opts_->max_split_,
                                       {.reassignment_in_flight = true}));
  }

  reassignment_set_.erase(token);

  const DatapointIndex p_size = datapoints_by_token[token].size();
  const DatapointIndex p1_size = datapoints_by_token.back().size();
  const DatapointIndex other_size = original_size - p_size - p1_size;
  VLOG(3) << "Split " << token << " / " << n_tokens
          << " Size: " << original_size << " -> " << p_size << " + " << p1_size
          << " + " << other_size;
  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::UpdateCentroid(DatapointPtr<float> x,
                                                    int32_t token,
                                                    bool stats_update) {
  Datapoint<float> old_centroid;
  if (stats_update) {
    CopyToDatapoint(centroid_->LeafCenters()[token], &old_centroid);
  }

  SCANN_RET_CHECK_NE(nullptr, centroid_)
      << "Incremental training must be enabled for AddCentroid.";
  SCANN_RET_CHECK_LT(token, searcher_->datapoints_by_token_.size())
      << "Incorrect token number: " << token;

  SCANN_ASSIGN_OR_RETURN(auto mutator, centroid_->LeafCenters().GetMutator());
  SCANN_RETURN_IF_ERROR(mutator->UpdateDatapoint(x, token));

  if (stats_update) {
    stats_collector().UpdatePartitionCentroid(token, x, old_centroid.ToPtr());
  }

  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::UpdateCentroid(DatapointPtr<float> x,
                                                    int32_t token) {
  return UpdateCentroid(x, token, true);
}

template <typename Searcher>
void TreeXHybridMutator<Searcher>::CheckReassignment(
    int32_t token, const MutationOptions& mo) {
  if (mo.reassignment_in_flight) {
    return;
  }

  if (token != kInvalidToken) mutation_stats_[token]++;

  const auto n = searcher_->datapoints_by_token_[token].size();
  if (n < incremental_opts_->cluster_stability_size_) return;

  const auto& threshold = incremental_opts_->incremental_threshold_;
  float real_threshold =
      std::holds_alternative<float>(threshold)
          ? (std::get<float>(incremental_opts_->incremental_threshold_) * n)
          : (std::get<DatapointIndex>(
                incremental_opts_->incremental_threshold_));
  if (mutation_stats_[token] <= std::ceil(real_threshold)) return;

  reassignment_set_.insert(token);
}

template <typename Searcher>
StatusOr<std::optional<ScannConfig>>
TreeXHybridMutator<Searcher>::IncrementalMaintenance() {
  if (searcher_->config().has_value() &&
      searcher_->config().value().has_autopilot()) {
    auto shared_dataset = searcher_->shared_dataset()
                              ? searcher_->shared_dataset()
                              : searcher_->reordering_helper().dataset();
    SCANN_ASSIGN_OR_RETURN(
        auto config, Autopilot(searcher_->config().value(), shared_dataset,
                               kInvalidDatapointIndex, kInvalidDimension));

    if (!config.has_partitioning()) return config;

    if (config.partitioning().has_incremental_training_config() &&
        config.partitioning().incremental_training_config().autopilot()) {
      int32_t num_partitions = config.partitioning().num_children();
      while (num_partitions < searcher_->datapoints_by_token_.size())
        SCANN_RETURN_IF_ERROR(ShrinkPartition());
      while (num_partitions > searcher_->datapoints_by_token_.size())
        SCANN_RETURN_IF_ERROR(SplitPartition());
      searcher_->set_config(config);
      auto query_spilling_max_centers =
          config.partitioning().query_spilling().max_spill_centers();
      centroid_->set_query_spilling_max_centers(query_spilling_max_centers);
    }
  }

  vector<int32_t> tokens(reassignment_set_.begin(), reassignment_set_.end());
  SCANN_RETURN_IF_ERROR(Reassign(tokens, {.reassignment_in_flight = true}));

  reassignment_set_.clear();
  return std::nullopt;
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::Reassign(ConstSpan<int32_t> tokens,
                                              const MutationOptions& mo) {
  vector<DatapointIndex> dptr_idx;
  DenseDataset<T> ds;
  Datapoint<T> dp;
  for (int32_t token : tokens) {
    VLOG(3) << "Reassign[" << token
            << "] size = " << searcher_->datapoints_by_token_[token].size()
            << " mutation = " << mutation_stats_[token];
    for (DatapointIndex i : searcher_->datapoints_by_token_[token]) {
      dptr_idx.push_back(i);
      SCANN_RETURN_IF_ERROR(ds.Append(GetDatapointPtr(i, &dp), ""));
    }

    mutation_stats_[token] = 0;
  }
  if (!ds.empty()) {
    auto precomputed = this->ComputePrecomputedMutationArtifacts(
        ds, this->mutation_threadpool());
    for (const auto& [i, idx] : Enumerate(dptr_idx)) {
      auto update_mo = mo;
      update_mo.precomputed_mutation_artifacts = precomputed[i].get();
      SCANN_RETURN_IF_ERROR(
          this->UpdateDatapoint(ds[i], idx, update_mo).status());
    }
  }
  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::ReassignTopK(int32_t token,
                                                  int32_t max_split,
                                                  const MutationOptions& mo) {
  auto n_tokens = searcher_->datapoints_by_token_.size();
  vector<uint32_t> perm;

  if (max_split == 2) {
    perm.push_back(token);
    perm.push_back(n_tokens - 1);
  } else {
    DenseDataset<T> ds;
    vector<DatapointIndex> dptr_idx(
        searcher_->datapoints_by_token_[token].begin(),
        searcher_->datapoints_by_token_[token].end());
    Datapoint<T> dp;
    for (DatapointIndex i : dptr_idx)
      SCANN_RETURN_IF_ERROR(ds.Append(GetDatapointPtr(i, &dp), ""));
    SCANN_ASSIGN_OR_RETURN(auto tokenization_result,
                           searcher_->TokenizeAndMaybeResidualize(ds));
    vector<DatapointIndex> count(n_tokens);
    for (const auto& token_res : tokenization_result)
      for (const auto& token : token_res.tokens) count[token]++;

    count[token] = std::numeric_limits<DatapointIndex>::max();
    count[n_tokens - 1] = std::numeric_limits<DatapointIndex>::max();
    perm.resize(count.size(), 0);
    std::iota(perm.begin(), perm.end(), 0);
    ZipSortBranchOptimized(std::greater<DatapointIndex>(), count.begin(),
                           count.end(), perm.begin(), perm.end());
  }

  DenseDataset<float> orig_centroid = centroid_->LeafCenters().Copy();

  vector<float> res(orig_centroid[0].dimensionality(),
                    std::numeric_limits<int32_t>::max());
  for (DatapointIndex i : Seq(n_tokens))
    SCANN_RETURN_IF_ERROR(UpdateCentroid(MakeDatapointPtr(res), i, false));
  for (DatapointIndex i : Seq(max_split))
    SCANN_RETURN_IF_ERROR(
        UpdateCentroid(orig_centroid[perm[i]], perm[i], false));

  SCANN_RETURN_IF_ERROR(Reassign(token, mo));

  const DenseDataset<float>& centroid = centroid_->LeafCenters();
  for (DatapointIndex i : Seq(max_split)) {
    auto to = orig_centroid.mutable_data(perm[i]);
    auto from = centroid[perm[i]].values_span();
    std::memcpy(to.data(), from.data(), from.size() * sizeof(float));
  }

  for (DatapointIndex i : Seq(n_tokens)) {
    SCANN_RETURN_IF_ERROR(UpdateCentroid(orig_centroid[i], i, false));
  }
  return OkStatus();
}

template <typename Searcher>
StatusOr<unique_ptr<TreeXHybridMutator<Searcher>>>
TreeXHybridMutator<Searcher>::Create(Searcher* searcher) {
  vector<typename SingleMachineSearcherBase<T>::Mutator*> leaf_mutators;
  auto leaf_searchers = MakeMutableSpan(searcher->leaf_searchers_);
  for (int i = 0; i < searcher->leaf_searchers_.size(); ++i) {
    SCANN_ASSIGN_OR_RETURN(auto leaf_mutator, leaf_searchers[i]->GetMutator());
    leaf_mutators.push_back(leaf_mutator);
  }

  decltype(global_to_local_) global_to_local;
  if (searcher->datapoints_by_token_disjoint_) {
    global_to_local = CreateGlobalToLocal<GlobalToLocal1>(
        MakeMutableSpan(searcher->datapoints_by_token_),
        searcher->num_datapoints_);
  } else {
    global_to_local = CreateGlobalToLocal<GlobalToLocal2>(
        MakeMutableSpan(searcher->datapoints_by_token_),
        searcher->num_datapoints_);
  }
  auto result = absl::WrapUnique(new TreeXHybridMutator<Searcher>(
      searcher, std::move(leaf_mutators), std::move(global_to_local)));
  SCANN_RETURN_IF_ERROR(result->PrepareForBaseMutation(searcher));
  DLOG(INFO) << "Initial disjointness = "
             << searcher->datapoints_by_token_disjoint_;

  if constexpr (std::is_same_v<Searcher, TreeAHHybridResidual>)
    searcher->leaf_tokens_by_norm_.resize(0);

  return std::move(result);
}

class TreeAHHybridResidual;
template <typename Searcher>
unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>
TreeXHybridMutator<Searcher>::ComputeLeafPrecomputedMutationArtifacts(
    int32_t token, const DatapointPtr<T>& maybe_residual,
    const DatapointPtr<T>& original) const {
  if (std::is_same<Searcher, TreeAHHybridResidual>::value) {
    return down_cast<typename asymmetric_hashing2::Searcher<T>::Mutator*>(
               leaf_mutators_[token])
        ->ComputePrecomputedMutationArtifacts(maybe_residual, original);
  } else {
    return leaf_mutators_[token]->ComputePrecomputedMutationArtifacts(
        maybe_residual);
  }
}

template <typename Searcher>
unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>
TreeXHybridMutator<Searcher>::ComputePrecomputedMutationArtifacts(
    const DatapointPtr<T>& dptr) const {
  auto tokenization_result_or_error =
      searcher_->TokenizeAndMaybeResidualize(dptr);
  if (!tokenization_result_or_error.status().ok()) {
    LOG_FIRST_N(WARNING, 10) << tokenization_result_or_error.status();
    return nullptr;
  }
  auto result = make_unique<TreeXPrecomputedMutationArtifacts>(
      std::move(tokenization_result_or_error.value()));
  result->leaf_precomputed_artifacts.resize(result->tokens.size());
  for (auto [token_idx, token] : Enumerate(result->tokens)) {
    result->leaf_precomputed_artifacts[token_idx] =
        ComputeLeafPrecomputedMutationArtifacts(
            token, result->GetMaybeResidual(dptr, token_idx), dptr);
  }
  return result;
}

template <typename Searcher>
vector<
    unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>>
TreeXHybridMutator<Searcher>::ComputePrecomputedMutationArtifacts(
    const TypedDataset<T>& batch) const {
  vector<unique_ptr<
      UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>>
      result_upcast(batch.size());

  auto tokenization_result_or_error =
      searcher_->TokenizeAndMaybeResidualize(batch);
  if (!tokenization_result_or_error.status().ok()) {
    LOG_FIRST_N(WARNING, 10) << tokenization_result_or_error.status();
    return result_upcast;
  }
  auto result = std::move(tokenization_result_or_error.value());
  for (size_t i : IndicesOf(result)) {
    result[i].leaf_precomputed_artifacts.resize(result[i].tokens.size());
    DatapointPtr<T> dptr = batch[i];
    for (auto [token_idx, token] : Enumerate(result[i].tokens)) {
      result[i].leaf_precomputed_artifacts[token_idx] =
          ComputeLeafPrecomputedMutationArtifacts(
              token, result[i].GetMaybeResidual(dptr, token_idx), dptr);
    }
  }

  for (size_t i : IndicesOf(result)) {
    result_upcast[i] =
        make_unique<TreeXPrecomputedMutationArtifacts>(std::move(result[i]));
  }
  return result_upcast;
}

template <typename Searcher>
vector<
    unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>>
TreeXHybridMutator<Searcher>::ComputePrecomputedMutationArtifacts(
    const TypedDataset<T>& ds, std::shared_ptr<ThreadPool> thread_pool) const {
  int n = ds.size();
  int num_threads = thread_pool ? thread_pool->NumThreads() : 1;
  size_t bs = DivRoundUp(n, num_threads);

  vector<unique_ptr<PrecomputedMutationArtifacts>> precomputed(n);
  ParallelFor<1>(
      Seq(DivRoundUp(n, bs)), thread_pool ? thread_pool.get() : nullptr,
      [&](size_t i) {
        size_t begin = bs * i;
        size_t size = std::min<DatapointIndex>(n - begin, bs);
        if (size <= 0) return;
        DenseDataset<T> batch;

        for (size_t j : Seq(size))
          QCHECK_OK(batch.Append(MakeDatapointPtr(
              ds[begin + j].values(), ds[begin + j].dimensionality())));
        auto batch_res = this->ComputePrecomputedMutationArtifacts(batch);
        for (size_t j : Seq(size))
          precomputed[begin + j] = std::move(batch_res[j]);
      });
  return precomputed;
}

template <typename Searcher>
StatusOr<DatapointIndex> TreeXHybridMutator<Searcher>::AddDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForAdd(dptr, docid, mo));
  DCHECK(searcher_);
  PrecomputedMutationArtifacts* ma = mo.precomputed_mutation_artifacts;
  unique_ptr<PrecomputedMutationArtifacts> ma_storage;
  if (!ma) {
    ma_storage = ComputePrecomputedMutationArtifacts(dptr);
    ma = ma_storage.get();
    SCANN_RET_CHECK(ma);
  }
  auto dc = dynamic_cast<TreeXPrecomputedMutationArtifacts*>(ma);
  if (!dc) {
    return InvalidArgumentError(
        absl::StrFormat("Invalid PrecomputedMutationArtifacts passed to "
                        "TreeXHybridMutator::AddDatapoint.  (Type = %s)",
                        typeid(*ma).name()));
  }
  SCANN_RETURN_IF_ERROR(UpgradeGlobalToLocalIfNeeded(dc->tokens.size()));
  return SCANN_DISPATCH_ON_GLOBAL_TO_LOCAL(AddDatapoint, dptr, docid, *dc, mo);
}

template <typename Searcher>
template <typename GlobalToLocal>
StatusOr<DatapointIndex> TreeXHybridMutator<Searcher>::AddDatapoint(
    const DatapointPtr<T>& dptr, string_view docid,
    const TreeXPrecomputedMutationArtifacts& ma, const MutationOptions& mo) {
  int32_t token_add = kInvalidToken;
  auto& global_to_local = std::get<GlobalToLocal>(global_to_local_);
  SCANN_ASSIGN_OR_RETURN(
      const DatapointIndex add_to_base_result,
      this->AddDatapointToBase(dptr, docid, MutateBaseOptions{}));
  if (add_to_base_result != kInvalidDatapointIndex) {
    SCANN_RET_CHECK_EQ(add_to_base_result, searcher_->num_datapoints_);
  }
  ConstSpan<int32_t> tokens = ma.tokens;
  if (tokens.size() > 1) {
    searcher_->datapoints_by_token_disjoint_ = false;
  }
  const DatapointIndex cur_dp_idx = searcher_->docids()->size() - 1;
  SCANN_RET_CHECK_EQ(global_to_local.size(), cur_dp_idx);
  global_to_local.emplace_back();
  auto& cur_global_to_local = global_to_local.back();
  SCANN_RET_CHECK_LE(tokens.size(), cur_global_to_local.size());
  InitializeGlobalToLocalEntry(cur_global_to_local);
  size_t cur_global_to_local_idx = 0;

  for (auto [token_idx, token] : Enumerate(tokens)) {
    SCANN_ASSIGN_OR_RETURN(
        const DatapointIndex local_dp_idx,
        leaf_mutators_[token]->AddDatapoint(
            ma.GetMaybeResidual(dptr, token_idx), "",
            MutationOptions{
                .precomputed_mutation_artifacts =
                    ma.leaf_precomputed_artifacts[token_idx].get()}));

    searcher_->datapoints_by_token_[token].push_back(cur_dp_idx);
    searcher_->leaf_size_upper_bound_ =
        std::max<uint32_t>(searcher_->leaf_size_upper_bound_,
                           searcher_->datapoints_by_token_[token].size());

    DCHECK_EQ(local_dp_idx, searcher_->datapoints_by_token_[token].size() - 1);
    cur_global_to_local[cur_global_to_local_idx++] = {token, local_dp_idx};

    if (token_add == kInvalidToken) token_add = token;
  }

  if (!mutation_stats_.empty() && token_add != kInvalidToken) {
    SCANN_RETURN_IF_ERROR(IngestUpdate(token_add, dptr, 1));
    CheckReassignment(token_add, mo);
  }
  stats_collector().AddStats(tokens, {add_to_base_result});
  ++searcher_->num_datapoints_;
  return searcher_->num_datapoints_ - 1;
}

template <typename Searcher>
void TreeXHybridMutator<Searcher>::Reserve(size_t size) {
  this->ReserveInBase(size);
  if (is_disjoint()) {
    std::get<GlobalToLocal1>(global_to_local_).reserve(size);
  } else {
    std::get<GlobalToLocal2>(global_to_local_).reserve(size);
  }
}

template <typename Searcher>
StatusOr<DatapointIndex> TreeXHybridMutator<Searcher>::UpdateDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  return UpdateDatapoint(dptr, index, mo);
}

template <typename Searcher>
StatusOr<DatapointIndex> TreeXHybridMutator<Searcher>::UpdateDatapoint(
    const DatapointPtr<T>& dptr, DatapointIndex index,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForUpdate(dptr, index, mo));
  DCHECK(searcher_);
  const bool mutate_values = true;
  vector<int32_t> tokens_storage;
  PrecomputedMutationArtifacts* ma = mo.precomputed_mutation_artifacts;
  unique_ptr<PrecomputedMutationArtifacts> ma_storage;
  if (!ma) {
    ma_storage = ComputePrecomputedMutationArtifacts(dptr);
    ma = ma_storage.get();
    SCANN_RET_CHECK(ma);
  }
  auto dc = dynamic_cast<TreeXPrecomputedMutationArtifacts*>(ma);
  if (!dc) {
    return InvalidArgumentError(
        absl::StrFormat("Invalid PrecomputedMutationArtifacts passed to "
                        "TreeXHybridMutator::AddDatapoint.  (Type = %s)",
                        typeid(*ma).name()));
  }
  SCANN_RETURN_IF_ERROR(UpgradeGlobalToLocalIfNeeded(dc->tokens.size()));
  return SCANN_DISPATCH_ON_GLOBAL_TO_LOCAL(UpdateDatapoint, dptr, index, *dc,
                                           mo);
}

template <typename Searcher>
template <typename GlobalToLocal>
StatusOr<DatapointIndex> TreeXHybridMutator<Searcher>::UpdateDatapoint(
    const DatapointPtr<T>& dptr, DatapointIndex dp_idx,
    const TreeXPrecomputedMutationArtifacts& ma, const MutationOptions& mo) {
  const bool mutate_values_vector = true;
  SCANN_RET_CHECK(mutate_values_vector);

  auto& global_to_local = std::get<GlobalToLocal>(global_to_local_);
  if (updated_token_centroid_ == kInvalidToken) {
    std::vector<int32_t> tmp_tokens;
    auto& g2l = global_to_local[dp_idx];
    tmp_tokens.reserve(g2l.size());
    for (auto [token, sub_index] : g2l) tmp_tokens.push_back(token);
    stats_collector().SubtractStats(tmp_tokens, {dp_idx});
  }

  Datapoint<T> orig;
  int32_t token_remove = kInvalidToken, token_add = kInvalidToken;
  if (!mutation_stats_.empty()) GetDatapointPtr(dp_idx, &orig, true);
  SCANN_RETURN_IF_ERROR(
      this->UpdateDatapointInBase(dptr, dp_idx, MutateBaseOptions{}));
  vector<int32_t> tokens = ma.tokens;
  if (tokens.size() > 1) {
    searcher_->datapoints_by_token_disjoint_ = false;
  }

  flat_hash_map<int, DatapointIndex> old_global_to_local;
  for (auto& elem : global_to_local[dp_idx]) {
    if (elem.first == kInvalidToken) continue;
    old_global_to_local[elem.first] = elem.second;
  }
  auto& new_global_to_local = global_to_local[dp_idx];
  SCANN_RET_CHECK_LE(tokens.size(), new_global_to_local.size());
  InitializeGlobalToLocalEntry(new_global_to_local);
  size_t new_global_to_local_idx = 0;

  MutableSpan<std::vector<DatapointIndex>> datapoints_by_token(
      searcher_->datapoints_by_token_);

  for (size_t token_idx : IndicesOf(tokens)) {
    int& token = tokens[token_idx];
    auto it = old_global_to_local.find(token);
    if (it == old_global_to_local.end()) continue;
    const size_t sub_index1 = it->second;
    const DatapointPtr<T> maybe_residualized =
        ma.GetMaybeResidual(dptr, token_idx);
    auto& leaf_precomputed_artifacts = ma.leaf_precomputed_artifacts[token_idx];

    SCANN_RETURN_IF_ERROR(
        leaf_mutators_[token]
            ->UpdateDatapoint(maybe_residualized, sub_index1,
                              MutationOptions{
                                  .precomputed_mutation_artifacts =
                                      leaf_precomputed_artifacts.get(),
                              })
            .status());

    if (token_add == kInvalidToken) token_add = token_remove = token;

    new_global_to_local[new_global_to_local_idx++] = {token, sub_index1};
    token = kInvalidToken;
    old_global_to_local.erase(it);
  }

  for (auto [token1, sub_index1] : old_global_to_local) {
    SCANN_RETURN_IF_ERROR(leaf_mutators_[token1]->RemoveDatapoint(sub_index1));
    if (sub_index1 == (datapoints_by_token[token1].size() - 1)) {
      datapoints_by_token[token1].pop_back();
    } else {
      const size_t index3 = datapoints_by_token[token1].back();
      auto status = UpdateSubIndex<GlobalToLocal>(index3, token1, sub_index1);
      if (!status.ok()) LOG(WARNING) << status;

      datapoints_by_token[token1].pop_back();
      datapoints_by_token[token1][sub_index1] = index3;
    }
    if (token_remove == kInvalidToken) token_remove = token1;
  }

  for (auto [token_idx, token] : Enumerate(tokens)) {
    if (token == kInvalidToken) continue;
    const DatapointPtr<T> maybe_residualized =
        ma.GetMaybeResidual(dptr, token_idx);
    auto& leaf_precomputed_artifacts = ma.leaf_precomputed_artifacts[token_idx];
    SCANN_ASSIGN_OR_RETURN(
        const DatapointIndex local_dp_idx,
        leaf_mutators_[token]->AddDatapoint(
            maybe_residualized, "",
            MutationOptions{.precomputed_mutation_artifacts =
                                leaf_precomputed_artifacts.get()}));

    datapoints_by_token[token].push_back(dp_idx);
    searcher_->leaf_size_upper_bound_ = std::max<uint32_t>(
        searcher_->leaf_size_upper_bound_, datapoints_by_token[token].size());
    DCHECK_EQ(local_dp_idx, datapoints_by_token[token].size() - 1);
    new_global_to_local[new_global_to_local_idx++] = {token, local_dp_idx};
    if (token_add == kInvalidToken) token_add = token;
  }

  if (!mutation_stats_.empty() && token_add != kInvalidToken) {
    SCANN_RETURN_IF_ERROR(IngestUpdate(token_add, dptr, 1));
    CheckReassignment(token_add, mo);

    if (token_remove != updated_token_centroid_) {
      SCANN_RETURN_IF_ERROR(IngestUpdate(token_remove, orig.ToPtr(), -1));
      CheckReassignment(token_remove, mo);
    }
  }
  stats_collector().AddStats(ma.tokens, {dp_idx});
  SCANN_RET_CHECK_LE(new_global_to_local_idx, new_global_to_local.size());
  return dp_idx;
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::RemoveDatapoint(string_view docid) {
  DatapointIndex dp_idx;
  if (!this->LookupDatapointIndex(docid, &dp_idx)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  SCANN_RETURN_IF_ERROR(RemoveDatapoint(dp_idx));
  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::RemoveDatapoint(DatapointIndex dp_idx) {
  SCANN_RETURN_IF_ERROR(this->ValidateForRemove(dp_idx));
  return SCANN_DISPATCH_ON_GLOBAL_TO_LOCAL(RemoveDatapointImpl, dp_idx);
}

template <typename Searcher>
template <typename GlobalToLocal>
Status TreeXHybridMutator<Searcher>::RemoveDatapointImpl(
    DatapointIndex dp_idx) {
  auto& global_to_local = std::get<GlobalToLocal>(global_to_local_);
  if (dp_idx >= global_to_local.size()) {
    return NotFoundError(absl::StrCat("Datapoint not found: ", dp_idx, "."));
  }

  const size_t old_size = searcher_->num_datapoints_;

  Datapoint<T> orig;
  int32_t token_remove = kInvalidToken;
  if (!mutation_stats_.empty()) GetDatapointPtr(dp_idx, &orig, true);

  std::vector<int32_t> tmp_tokens;
  auto& g2l = global_to_local[dp_idx];
  tmp_tokens.reserve(g2l.size());
  for (auto [token, sub_index] : g2l) tmp_tokens.push_back(token);
  stats_collector().SubtractStats(tmp_tokens, {dp_idx});

  SCANN_RETURN_IF_ERROR(this->RemoveDatapointFromBase(dp_idx).status());

  MutableSpan<std::vector<DatapointIndex>> datapoints_by_token(
      searcher_->datapoints_by_token_);

  for (auto [token, sub_index] : global_to_local[dp_idx]) {
    if (token == kInvalidToken) continue;
    SCANN_RETURN_IF_ERROR(leaf_mutators_[token]->RemoveDatapoint(sub_index))
        << dp_idx << " " << token << " " << sub_index;

    if (dp_idx == old_size - 1) {
      if (sub_index == (datapoints_by_token[token].size() - 1)) {
      } else {
        auto status = UpdateSubIndex<GlobalToLocal>(
            datapoints_by_token[token].back(), token, sub_index);
        if (!status.ok()) LOG(WARNING) << status;

        datapoints_by_token[token][sub_index] =
            datapoints_by_token[token].back();
      }
    } else {
      for (auto [token2, sub_index2] : global_to_local[old_size - 1]) {
        if (token2 == kInvalidToken) continue;
        if (sub_index == datapoints_by_token[token].size() - 1) {
          datapoints_by_token[token2][sub_index2] = dp_idx;
        } else if (token2 == token &&
                   sub_index2 == datapoints_by_token[token].size() - 1) {
          sub_index2 = sub_index;
          auto status =
              UpdateSubIndex<GlobalToLocal>(old_size - 1, token2, sub_index);
          if (!status.ok()) LOG(WARNING) << status;
          DCHECK_EQ(datapoints_by_token[token][sub_index], dp_idx);
        } else {
          const size_t index3 = datapoints_by_token[token].back();
          if (index3 == old_size - 1) {
            DCHECK(!is_disjoint());
            DCHECK_EQ(datapoints_by_token[token][sub_index], dp_idx);
          } else {
            auto status =
                UpdateSubIndex<GlobalToLocal>(index3, token, sub_index);
            if (!status.ok()) LOG(WARNING) << status;

            datapoints_by_token[token][sub_index] = index3;
          }

          datapoints_by_token[token2][sub_index2] = dp_idx;
        }
      }
    }

    if (token_remove == kInvalidToken) token_remove = token;
    datapoints_by_token[token].pop_back();
  }

  if (dp_idx != old_size - 1) {
    global_to_local[dp_idx] = global_to_local[old_size - 1];
  }
  global_to_local.pop_back();
  --searcher_->num_datapoints_;
  this->OnDatapointIndexRename(old_size - 1, dp_idx);

  if (!mutation_stats_.empty() && token_remove != kInvalidToken) {
    SCANN_RETURN_IF_ERROR(IngestUpdate(token_remove, orig.ToPtr(), -1));
    CheckReassignment(token_remove);
  }
  return OkStatus();
}

template <typename Searcher>
Status TreeXHybridMutator<Searcher>::CheckGlobalToLocalConsistency() const {
  auto docids = searcher_->docids();
  if (docids) {
    SCANN_RET_CHECK_EQ(searcher_->num_datapoints_, docids->size());
  }
  vector<flat_hash_map<int, DatapointIndex>> rebuilt(
      searcher_->num_datapoints_);
  for (int token_idx : IndicesOf(searcher_->datapoints_by_token_)) {
    for (auto [sub_idx, global_idx] :
         Enumerate(searcher_->datapoints_by_token_[token_idx])) {
      SCANN_RET_CHECK_LT(global_idx, rebuilt.size());
      SCANN_RET_CHECK(rebuilt[global_idx].insert({token_idx, sub_idx}).second)
          << "Duplicate token idx in rebuilt:  " << token_idx;
    }
  }

  auto impl = [&](auto& global_to_local) -> Status {
    for (DatapointIndex global_idx : IndicesOf(global_to_local)) {
      auto& rebuilt_map = rebuilt[global_idx];
      SCANN_RET_CHECK_LE(rebuilt_map.size(), 2);
      auto original_vec = MakeConstSpan(global_to_local[global_idx]);
      while (!original_vec.empty() &&
             original_vec.back().first == kInvalidToken) {
        original_vec.remove_suffix(1);
      }
      SCANN_RET_CHECK_GE(original_vec.size(), 1);
      for (auto [token_idx, sub_idx] : original_vec) {
        auto it = rebuilt_map.find(token_idx);
        SCANN_RET_CHECK(it != rebuilt_map.end())
            << "Token idx found in original but not rebuilt:  " << token_idx;
        SCANN_RET_CHECK_EQ(sub_idx, it->second)
            << "subidx mismatch for token " << token_idx;
      }
      if (original_vec.size() == rebuilt_map.size()) continue;
      flat_hash_map<int, DatapointIndex> original_map;
      original_map.reserve(original_vec.size());
      for (auto [token_idx, sub_idx] : original_vec) {
        SCANN_RET_CHECK(original_map.insert({token_idx, sub_idx}).second)
            << "Duplicate token idx in global_to_local:  " << token_idx;
      }
      for (auto [token_idx, sub_idx] : rebuilt_map) {
        SCANN_RET_CHECK(original_map.contains(token_idx))
            << "Token idx found in rebuilt but not original:  " << token_idx;
      }
    }
    return OkStatus();
  };
  if (is_disjoint()) {
    SCANN_RET_CHECK(searcher_->datapoints_by_token_disjoint_);
    return impl(std::get<GlobalToLocal1>(global_to_local_));
  } else {
    SCANN_RET_CHECK(!searcher_->datapoints_by_token_disjoint_);
    return impl(std::get<GlobalToLocal2>(global_to_local_));
  }
}

#undef SCANN_DISPATCH_ON_GLOBAL_TO_LOCAL

}  // namespace research_scann

#endif
