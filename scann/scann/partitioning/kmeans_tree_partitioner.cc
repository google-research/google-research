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

#include "scann/partitioning/kmeans_tree_partitioner.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/internal/spinlock.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/kmeans_tree_partitioner.pb.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/trees/kmeans_tree/kmeans_tree_node.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

template <typename T>
KMeansTreePartitioner<T>::KMeansTreePartitioner(
    shared_ptr<const DistanceMeasure> database_tokenization_dist,
    shared_ptr<const DistanceMeasure> query_tokenization_dist)
    : database_tokenization_dist_(database_tokenization_dist),
      query_tokenization_dist_(query_tokenization_dist) {}

template <typename T>
KMeansTreePartitioner<T>::KMeansTreePartitioner(
    shared_ptr<const DistanceMeasure> database_tokenization_dist,
    shared_ptr<const DistanceMeasure> query_tokenization_dist,
    const SerializedKMeansTreePartitioner& proto)
    : kmeans_tree_(make_shared<KMeansTree>(proto.kmeans_tree())),
      database_tokenization_dist_(database_tokenization_dist),
      query_tokenization_dist_(query_tokenization_dist) {}

template <typename T>
KMeansTreePartitioner<T>::KMeansTreePartitioner(
    shared_ptr<const DistanceMeasure> database_tokenization_dist,
    shared_ptr<const DistanceMeasure> query_tokenization_dist,
    shared_ptr<const KMeansTree> pretrained_tree)
    : kmeans_tree_(std::move(pretrained_tree)),
      database_tokenization_dist_(database_tokenization_dist),
      query_tokenization_dist_(query_tokenization_dist) {
  CHECK(kmeans_tree_->is_trained())
      << "The pre-trained tree overload of KMeansTreePartitioner can only be "
         "used with a tree that has already been trained.";
}

template <typename T>
unique_ptr<Partitioner<T>> KMeansTreePartitioner<T>::Clone() const {
  auto result = make_unique<KMeansTreePartitioner<T>>(
      database_tokenization_dist_, query_tokenization_dist_, kmeans_tree_);
  result->query_spilling_type_ = query_spilling_type_;
  result->query_spilling_threshold_ = query_spilling_threshold_;
  result->query_spilling_max_centers_ = query_spilling_max_centers_;
  result->query_tokenization_type_ = query_tokenization_type_;
  result->database_tokenization_type_ = database_tokenization_type_;
  result->database_tokenization_searcher_ = database_tokenization_searcher_;
  result->database_spilling_fixed_number_of_centers_ =
      database_spilling_fixed_number_of_centers_;
  result->orthogonality_amplification_lambda_ =
      orthogonality_amplification_lambda_;
  result->query_tokenization_searcher_ = query_tokenization_searcher_;
  result->num_tokenized_branch_ = num_tokenized_branch_;
  return std::move(result);
}

template <typename T>
KMeansTreePartitioner<T>::~KMeansTreePartitioner() {}

template <typename T>
Status KMeansTreePartitioner<T>::CreatePartitioning(
    const Dataset& training_dataset, const DistanceMeasure& training_dist,
    int32_t k_per_level, KMeansTreeTrainingOptions* opts) {
  if (kmeans_tree_) {
    return FailedPreconditionError(
        "Cannot call CreatePartitioning twice with the same "
        "KMeansTreePartitioner.");
  }
  auto tree = make_shared<KMeansTree>();
  SCANN_RETURN_IF_ERROR(
      tree->Train(training_dataset, training_dist, k_per_level, opts));
  kmeans_tree_ = std::move(tree);
  return OkStatus();
}

template <typename T>
void KMeansTreePartitioner<T>::set_query_spilling_threshold(double val) {
  query_spilling_threshold_ = val;
}

constexpr int kAhMultiplierSpilling = 10;

constexpr int kAhMinReorderedPartitions = 100;

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, pair<DatapointIndex, float>* result) const {
  DCHECK(result);
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Cannot query a KMeansTreePartitioner before training.");
  }

  const TokenizationType cur_type = cur_tokenization_type();
  const bool is_query_mode =
      this->tokenization_mode() == UntypedPartitioner::QUERY;
  if (cur_type == ASYMMETRIC_HASHING) {
    int pre_reordering_num_neighbors =
        TokenizationSearcher()->reordering_enabled() ? kAhMinReorderedPartitions
                                                     : 1;
    return TokenForDatapointUseSearcher(dptr, result,
                                        pre_reordering_num_neighbors);
  } else {
    vector<pair<DatapointIndex, float>> result_vec;
    const shared_ptr<const DistanceMeasure>& dist =
        is_query_mode ? query_tokenization_dist_ : database_tokenization_dist_;

    auto tokenization_options = KMeansTree::TokenizationOptions::NoSpilling(
        static_cast<KMeansTree::TokenizationType>(cur_type));
    tokenization_options.num_tokenized_branch = num_tokenized_branch_;
    SCANN_RETURN_IF_ERROR(
        kmeans_tree_->Tokenize(dptr, *dist, tokenization_options, &result_vec));
    *result = result_vec[0];
    return OkStatus();
  }
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapointBatched(
    const TypedDataset<T>& queries, vector<int32_t>* results,
    ThreadPool* pool) const {
  if (cur_tokenization_type() != FLOAT || queries.IsSparse() ||
      !kmeans_tree_->is_flat()) {
    return Partitioner<T>::TokenForDatapointBatched(queries, results);
  }
  SCANN_ASSIGN_OR_RETURN(auto top1_results,
                         TokenForDatapointBatchedImpl(queries, pool));
  results->resize(queries.size());
  for (size_t j : Seq(queries.size())) {
    (*results)[j] = static_cast<int32_t>(top1_results[j].first);
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapointBatched(
    const TypedDataset<T>& queries,
    vector<pair<DatapointIndex, float>>* results, ThreadPool* pool) const {
  if (cur_tokenization_type() != FLOAT || queries.IsSparse() ||
      !kmeans_tree_->is_flat()) {
    results->resize(queries.size());
    for (size_t i : IndicesOf(queries)) {
      SCANN_RETURN_IF_ERROR(TokenForDatapoint(queries[i], &results->at(i)));
    }
  }
  SCANN_ASSIGN_OR_RETURN(*results, TokenForDatapointBatchedImpl(queries, pool));
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<pair<DatapointIndex, float>>* result) const {
  DCHECK(result);

  if (this->tokenization_mode() == UntypedPartitioner::QUERY) {
    const auto max_centers = max_centers_override > 0
                                 ? max_centers_override
                                 : query_spilling_max_centers_;

    if (query_tokenization_type_ == ASYMMETRIC_HASHING) {
      int pre_reordering_num_neighbors =
          TokenizationSearcher()->reordering_enabled()
              ? std::max(kAhMinReorderedPartitions,
                         max_centers * kAhMultiplierSpilling)
              : max_centers;
      return TokensForDatapointWithSpillingUseSearcher(
          dptr, result, max_centers, pre_reordering_num_neighbors);
    }

    return kmeans_tree_->Tokenize(
        dptr, *query_tokenization_dist_,
        KMeansTree::TokenizationOptions::UserSpecifiedSpilling(
            query_spilling_type_, query_spilling_threshold_, max_centers,
            static_cast<KMeansTree::TokenizationType>(
                query_tokenization_type_)),
        result);
  } else if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    if (orthogonality_amplified_database_spilling()) {
      if (!dptr.IsDense()) {
        return UnimplementedError(
            "Orthogonality amplification isn't implemented for sparse data.");
      }
      result->resize(2);
      SCANN_RETURN_IF_ERROR(TokenForDatapoint(dptr, &result->front()));

      DenseDataset<T> ds;
      ds.AppendOrDie(dptr, "");
      SCANN_RETURN_IF_ERROR(OrthogonalityAmplifiedTokenForDatapointBatched(
          ds, MakeConstSpan(*result).subspan(0, 1),
          MakeMutableSpan(*result).subspan(1, 1)));
      if (result->at(0).first == result->at(1).first) {
        result->resize(1);
      }
      return OkStatus();
    }

    if (database_spilling_fixed_number_of_centers_ > 0) {
      if (database_tokenization_type_ == ASYMMETRIC_HASHING) {
        int pre_reordering_num_neighbors =
            TokenizationSearcher()->reordering_enabled()
                ? std::max(kAhMinReorderedPartitions,
                           database_spilling_fixed_number_of_centers_ *
                               kAhMultiplierSpilling)
                : database_spilling_fixed_number_of_centers_;
        return TokensForDatapointWithSpillingUseSearcher(
            dptr, result, database_spilling_fixed_number_of_centers_,
            pre_reordering_num_neighbors);
      } else {
        return kmeans_tree_->Tokenize(
            dptr, *query_tokenization_dist_,
            KMeansTree::TokenizationOptions::UserSpecifiedSpilling(
                QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS, 0.0,
                database_spilling_fixed_number_of_centers_,
                static_cast<KMeansTree::TokenizationType>(
                    query_tokenization_type_)),
            result);
      }
    }

    if (database_tokenization_type_ == ASYMMETRIC_HASHING) {
      if (kmeans_tree_->learned_spilling_type() ==
          DatabaseSpillingConfig::NO_SPILLING) {
        result->resize(1);
        SCANN_RETURN_IF_ERROR(TokenForDatapoint(dptr, &result->front()));
        return OkStatus();

      } else {
        return FailedPreconditionError(
            "ASYMMETRIC_HASHING database tokenization with spilling does not "
            "support spilling_type other than NO_SPILLING and "
            "FIXED_NUMBER_OF_CENTERS.");
      }
    }

    return kmeans_tree_->Tokenize(
        dptr, *database_tokenization_dist_,
        KMeansTree::TokenizationOptions::LearnedSpilling(
            static_cast<KMeansTree::TokenizationType>(
                database_tokenization_type_)),
        result);
  } else {
    return InternalError(absl::StrCat("Unknown tokenization mode:  ",
                                      this->tokenization_mode()));
  }
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapointUseSearcher(
    const DatapointPtr<T>& dptr, pair<DatapointIndex, float>* result,
    int32_t pre_reordering_num_neighbors) const {
  if (!TokenizationSearcher()) {
    return FailedPreconditionError(
        "CreateAsymmetricHashingSearcherForTokenization must "
        "be called first.");
  }

  DCHECK(kmeans_tree_->is_flat());

  Datapoint<float> dp;
  DatapointPtr<float> query = ToFloat(dptr, &dp);

  SearchParameters params(pre_reordering_num_neighbors,
                          numeric_limits<float>::infinity(), 1,
                          numeric_limits<float>::infinity());

  NNResultsVector search_result;
  Status status =
      TokenizationSearcher()->FindNeighbors(query, params, &search_result);
  if (!status.ok()) return status;

  DCHECK_LE(search_result[0].first, numeric_limits<int32_t>::max());
  *result = search_result.front();
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapoint(const DatapointPtr<T>& dptr,
                                                   int32_t* result) const {
  pair<DatapointIndex, float> res_pair;
  SCANN_RETURN_IF_ERROR(TokenForDatapoint(dptr, &res_pair));
  *result = res_pair.first;
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingAndOverride(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<int32_t>* result) const {
  vector<pair<DatapointIndex, float>> result_raw;
  SCANN_RETURN_IF_ERROR(
      TokensForDatapointWithSpilling(dptr, max_centers_override, &result_raw));
  result->clear();
  result->reserve(result_raw.size());
  for (auto& elem : result_raw) {
    result->push_back(elem.first);
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingUseSearcher(
    const DatapointPtr<T>& dptr, vector<pair<DatapointIndex, float>>* result,
    int32_t num_neighbors, int32_t pre_reordering_num_neighbors) const {
  if (!TokenizationSearcher()) {
    return FailedPreconditionError(
        "CreateAsymmetricHashingSearcherForTokenization must "
        "be called first.");
  }
  if (orthogonality_amplified_database_spilling()) {
    return UnimplementedError(
        "Orthogonality amplification isn't implemented with searcher-based "
        "partitioning.");
  }

  Datapoint<float> dp;
  DatapointPtr<float> query = ToFloat(dptr, &dp);
  float threshold = numeric_limits<float>::infinity();
  if (query_spilling_type_ == QuerySpillingConfig::ABSOLUTE_DISTANCE) {
    threshold = query_spilling_threshold_;
  }
  SearchParameters params(pre_reordering_num_neighbors,
                          numeric_limits<float>::infinity(), num_neighbors,
                          threshold);
  DCHECK(kmeans_tree_->is_flat());
  return TokenizationSearcher()->FindNeighbors(query, params, result);
}

namespace {

template <typename FloatT, typename T>
Datapoint<FloatT> ResidualizeImpl(const DatapointPtr<T>& dptr,
                                  const DatapointPtr<float>& center,
                                  float multiplier = 1.0) {
  Datapoint<FloatT> residual;
  auto& values = *residual.mutable_values();
  values.resize(center.nonzero_entries());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = (static_cast<float>(dptr.values()[i]) - center.values()[i]) *
                multiplier;
  }
  return residual;
}

}  // namespace

template <typename T>
StatusOr<Datapoint<float>> KMeansTreePartitioner<T>::ResidualizeToFloat(
    const DatapointPtr<T>& dptr, int32_t token) const {
  const DatapointPtr<float> center = kmeans_tree()->is_flat()
                                         ? LeafCenters()[token]
                                         : kmeans_tree()->CenterForToken(token);
  return ResidualizeImpl<float>(dptr, center);
}

template <typename T>
const DenseDataset<float>& KMeansTreePartitioner<T>::LeafCenters() const {
  if (kmeans_tree_->is_flat()) return kmeans_tree_->root()->Centers();
  {
    absl::ReaderMutexLock lock(&leaf_centers_mutex_);
    if (!leaf_centers_.empty()) return leaf_centers_;
  }
  absl::MutexLock lock(&leaf_centers_mutex_);
  if (!leaf_centers_.empty()) return leaf_centers_;
  std::function<void(const KMeansTreeNode&)> recurse_lambda =
      [&](const KMeansTreeNode& node)
          ABSL_EXCLUSIVE_LOCKS_REQUIRED(leaf_centers_mutex_) {
            if (node.IsLeaf()) {
              if (leaf_centers_.empty()) {
                leaf_centers_.set_dimensionality(
                    node.cur_node_center().dimensionality());
                leaf_centers_.Reserve(n_tokens());
              }
              CHECK_EQ(node.LeafId(), leaf_centers_.size());
              leaf_centers_.AppendOrDie(node.cur_node_center(), "");
            } else {
              for (const auto& child : node.Children()) {
                recurse_lambda(child);
              }
            }
          };
  recurse_lambda(*kmeans_tree_->root());
  return leaf_centers_;
}

template <typename T>
Status KMeansTreePartitioner<T>::ApplyAvq(
    const DenseDataset<T>& dataset,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token, float avq_eta,
    ThreadPool* pool_or_null) {
  if (!kmeans_tree_.unique()) {
    return FailedPreconditionError(
        "Cannot apply AVQ to KMeansTreePartitioner instances with a shared "
        "KMeansTree.");
  }

  SCANN_RETURN_IF_ERROR(
      const_cast<KMeansTree*>(kmeans_tree_.get())
          ->ApplyAvq(dataset, datapoints_by_token, avq_eta, pool_or_null));
  absl::MutexLock lock(&leaf_centers_mutex_);
  leaf_centers_ = DenseDataset<float>();
  return OkStatus();
}

template <typename T>
void KMeansTreePartitioner<T>::CopyToProto(
    SerializedPartitioner* result) const {
  DCHECK(result);
  DCHECK(kmeans_tree_)
      << "Cannot call CopyToProto on a KMeansTreePartitioner that has not "
      << "been successfully trained.";
  result->Clear();
  result->set_n_tokens(n_tokens());
  auto kmeans_proto = result->mutable_kmeans();

  kmeans_tree_->SerializeWithoutIndices(kmeans_proto->mutable_kmeans_tree());
}

namespace {

template <typename ResultType, typename T>
DenseDataset<ResultType> GetBatchSubmatrix(const DenseDataset<T>& database,
                                           size_t start, size_t end) {
  DCHECK(!database.is_binary());
  DCHECK_GT(end, start);
  const size_t length = end - start;
  vector<ResultType> storage(length * database.dimensionality());
  auto base_ptr = database[start].values();
  for (size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<ResultType>(base_ptr[i]);
  }
  return DenseDataset<ResultType>(std::move(storage), length);
}

}  // namespace

template <typename T>
StatusOr<vector<std::vector<DatapointIndex>>>
KMeansTreePartitioner<T>::TokenizeDatabase(const TypedDataset<T>& database,
                                           ThreadPool* pool_or_null) const {
  return const_cast<KMeansTreePartitioner<T>*>(this)->TokenizeDatabase(
      database, pool_or_null, AvqOptions{});
}

template <typename T>
StatusOr<vector<std::vector<DatapointIndex>>>
KMeansTreePartitioner<T>::TokenizeDatabase(const TypedDataset<T>& database,
                                           ThreadPool* pool_or_null,
                                           AvqOptions avq_opts) {
  if (this->tokenization_mode() != UntypedPartitioner::DATABASE) {
    return FailedPreconditionError(
        "Cannot run TokenizeDatabase when not in database tokenization mode.");
  }
  if (avq_opts.avq_after_primary && !database.IsDense()) {
    return UnimplementedError("AVQ is not supported with sparse databases.");
  }
  auto dense = [&database]() -> const DenseDataset<T>& {
    CHECK(database.IsDense());
    return *down_cast<const DenseDataset<T>*>(&database);
  };
  if (orthogonality_amplified_database_spilling()) {
    if (!database.IsDense()) {
      return UnimplementedError(
          "Orthogonality amplification only works with dense data.");
    }
    vector<pair<DatapointIndex, float>> primary_results;
    SCANN_RETURN_IF_ERROR(
        TokenForDatapointBatched(database, &primary_results, pool_or_null));
    vector<std::vector<DatapointIndex>> token_to_datapoint_index(
        this->n_tokens());
    for (auto [dp_idx, primary] : Enumerate(primary_results)) {
      const int32_t token = primary.first;
      SCANN_RET_CHECK_LT(token, token_to_datapoint_index.size());
      token_to_datapoint_index[token].push_back(dp_idx);
    }
    if (avq_opts.avq_after_primary) {
      SCANN_RETURN_IF_ERROR(ApplyAvq(dense(), token_to_datapoint_index,
                                     avq_opts.avq_eta, pool_or_null));
    }
    if (avq_opts.skip_secondary_tokenization) {
      return token_to_datapoint_index;
    }
    vector<pair<DatapointIndex, float>> secondary_results(
        primary_results.size());
    SCANN_RETURN_IF_ERROR(OrthogonalityAmplifiedTokenForDatapointBatched(
        *down_cast<const DenseDataset<T>*>(&database), primary_results,
        MakeMutableSpan(secondary_results), pool_or_null));
    for (auto [dp_idx, secondary] : Enumerate(secondary_results)) {
      const int32_t token = secondary.first;
      SCANN_RET_CHECK_LT(token, token_to_datapoint_index.size());
      if (token == primary_results[dp_idx].first) continue;
      token_to_datapoint_index[token].push_back(dp_idx);
    }
    for (auto& elem : token_to_datapoint_index) {
      elem.shrink_to_fit();
      std::sort(elem.begin(), elem.end());
    }
    return token_to_datapoint_index;
  } else if (typeid(*database_tokenization_dist_) ==
                 typeid(const SquaredL2Distance) &&
             kmeans_tree_->is_flat() && database.IsDense() &&
             kmeans_tree_->learned_spilling_type() ==
                 DatabaseSpillingConfig::NO_SPILLING &&
             (IsSame<T, float>() || IsSame<T, double>()) &&
             (database_tokenization_type_ == FLOAT)) {
    SCANN_ASSIGN_OR_RETURN(auto datapoint_index_to_result,
                           TokenizeDatabaseImplFastPath(dense(), pool_or_null));
    vector<std::vector<DatapointIndex>> token_to_datapoint_index(
        this->n_tokens());
    for (DatapointIndex dp_index : IndicesOf(datapoint_index_to_result)) {
      const int32_t token = datapoint_index_to_result[dp_index].first;
      token_to_datapoint_index[token].push_back(dp_index);
    }
    for (auto& elem : token_to_datapoint_index) {
      elem.shrink_to_fit();
    }
    if (avq_opts.avq_after_primary) {
      SCANN_RETURN_IF_ERROR(ApplyAvq(dense(), token_to_datapoint_index,
                                     avq_opts.avq_eta, pool_or_null));
    }
    return std::move(token_to_datapoint_index);
  } else {
    SCANN_ASSIGN_OR_RETURN(
        auto result, Partitioner<T>::TokenizeDatabase(database, pool_or_null));
    if (avq_opts.avq_after_primary) {
      SCANN_RETURN_IF_ERROR(ApplyAvq(dense(), result, avq_opts.avq_eta));
    }
    return result;
  }
}

template <typename T>
StatusOr<vector<pair<DatapointIndex, float>>>
KMeansTreePartitioner<T>::TokenizeDatabaseImplFastPath(
    const DenseDataset<T>& database, ThreadPool* pool_or_null) const {
  vector<pair<DatapointIndex, float>> datapoint_index_to_result;
  if (kmeans_tree_->root()->IsLeaf()) {
    datapoint_index_to_result.resize(database.size(), {0, NAN});
    return std::move(datapoint_index_to_result);
  }

  DCHECK_EQ(database_tokenization_type_, FLOAT);
  SCANN_ASSIGN_OR_RETURN(
      datapoint_index_to_result,
      TokenizeDatabaseImplFastPath(database, kmeans_tree_->root()->Centers(),
                                   pool_or_null));
  return std::move(datapoint_index_to_result);
}

template <typename T>
StatusOr<vector<pair<DatapointIndex, float>>>
KMeansTreePartitioner<T>::TokenizeDatabaseImplFastPath(
    const DenseDataset<T>& database, const DenseDataset<float>& centers,
    ThreadPool* pool_or_null) const {
  constexpr size_t kBatchSize = 128;
  vector<pair<DatapointIndex, float>> nearest_centers(database.size());
  SquaredL2Distance dist;

  ParallelFor<1>(
      SeqWithStride<kBatchSize>(0, database.size()), pool_or_null,
      [&](size_t batch_begin) {
        const size_t batch_end =
            std::min<size_t>(batch_begin + kBatchSize, database.size());
        DenseDataset<float> batch_submatrix =
            GetBatchSubmatrix<float>(database, batch_begin, batch_end);
        auto local_results =
            DenseDistanceManyToManyTop1<float>(dist, batch_submatrix, centers);
        std::copy(local_results.begin(), local_results.end(),
                  nearest_centers.begin() + batch_begin);
      });

  return nearest_centers;
}

template <>
StatusOr<vector<pair<DatapointIndex, float>>>
KMeansTreePartitioner<float>::TokenizeDatabaseImplFastPath(
    const DenseDataset<float>& database, const DenseDataset<float>& centers,
    ThreadPool* pool_or_null) const {
  return DenseDistanceManyToManyTop1<float>(SquaredL2Distance(), database,
                                            centers, pool_or_null);
}

template <typename T>
Status
KMeansTreePartitioner<T>::TokensForDatapointWithSpillingBatchedAndOverride(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<vector<int32_t>> results, ThreadPool* pool) const {
  vector<vector<pair<DatapointIndex, float>>> raw_results(queries.size());
  SCANN_RETURN_IF_ERROR(TokensForDatapointWithSpillingBatched(
      queries, max_centers_override, MakeMutableSpan(raw_results), pool));
  for (size_t i = 0; i < results.size(); ++i) {
    vector<int32_t>& cur_results = results[i];
    auto& cur_raw_results = raw_results[i];
    cur_results.clear();
    cur_results.reserve(cur_raw_results.size());
    for (auto& token_and_distance : cur_raw_results) {
      cur_results.push_back(token_and_distance.first);
    }
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<vector<pair<DatapointIndex, float>>> results,
    ThreadPool* pool) const {
  if (!max_centers_override.empty() &&
      queries.size() != max_centers_override.size()) {
    return InvalidArgumentError(
        "The max_centers override must have the same "
        "size as batched queries.");
  }

  auto fallback = [&]() -> Status {
    for (DatapointIndex i : IndicesOf(queries)) {
      const int32_t max_centers =
          max_centers_override.empty() ? 0 : max_centers_override[i];
      SCANN_RETURN_IF_ERROR(
          TokensForDatapointWithSpilling(queries[i], max_centers, &results[i]));
    }
    return OkStatus();
  };

  if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    if (orthogonality_amplified_database_spilling()) {
      if (!queries.IsDense()) {
        return UnimplementedError(
            "Orthogonality amplification only works with dense data.");
      }
      vector<pair<DatapointIndex, float>> primary_results;
      SCANN_RETURN_IF_ERROR(
          TokenForDatapointBatched(queries, &primary_results, pool));
      vector<pair<DatapointIndex, float>> secondary_results(results.size());
      SCANN_RETURN_IF_ERROR(OrthogonalityAmplifiedTokenForDatapointBatched(
          *down_cast<const DenseDataset<T>*>(&queries), primary_results,
          MakeMutableSpan(secondary_results), pool));
      for (size_t i : IndicesOf(primary_results)) {
        results[i] = {primary_results[i]};
        if (primary_results[i].first != secondary_results[i].first) {
          results[i].push_back(secondary_results[i]);
        }
      }
      return OkStatus();
    } else if (kmeans_tree_->learned_spilling_type() ==
                   DatabaseSpillingConfig::NO_SPILLING &&
               database_spilling_fixed_number_of_centers_ == 0) {
      vector<pair<DatapointIndex, float>> primary_results;
      SCANN_RETURN_IF_ERROR(
          TokenForDatapointBatched(queries, &primary_results, pool));
      for (size_t i : IndicesOf(primary_results)) {
        results[i] = {primary_results[i]};
      }
      return OkStatus();
    } else {
      return fallback();
    }
  } else if (!SupportsLowLevelQueryBatching() || !queries.IsDense()) {
    return fallback();
  }

  const DenseDataset<float>& centers = kmeans_tree_->root()->Centers();
  if (centers.dimensionality() != queries.dimensionality()) {
    return FailedPreconditionError(
        "Incorrect query dimensionality.  Expected %d, got %d.\n",
        centers.dimensionality(), queries.dimensionality());
  }

  DenseDataset<float> float_query_storage;
  auto float_queries = ConvertToFloatIfNecessary(
      *down_cast<const DenseDataset<T>*>(&queries), &float_query_storage);

  if (query_spilling_type_ == QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    vector<FastTopNeighbors<float>> ftns(float_queries->size());
    for (DatapointIndex query_idx : IndicesOf(*float_queries)) {
      const auto max_centers = max_centers_override.empty()
                                   ? query_spilling_max_centers_
                                   : max_centers_override[query_idx];
      ftns[query_idx] = FastTopNeighbors<float>(max_centers);
    }
    DenseDistanceManyToManyTopK(*query_tokenization_dist_, *float_queries,
                                centers, MakeMutableSpan(ftns));
    for (DatapointIndex query_idx : IndicesOf(*float_queries)) {
      ftns[query_idx].FinishUnsorted(&results[query_idx]);
      ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                                   ftns[query_idx].max_results() - 1,
                                   results[query_idx].begin(),
                                   results[query_idx].end());
    }
    return OkStatus();
  }

  vector<std::vector<float>> distance_mat(queries.size());
  for (auto& elem : distance_mat) {
    elem.resize(centers.size());
  }
  auto distance_callback = [&distance_mat](MutableSpan<float> dists,
                                           DatapointIndex base_dp_idx,
                                           DatapointIndex query_idx) {
    std::copy(dists.begin(), dists.end(),
              distance_mat[query_idx].begin() + base_dp_idx);
  };
  DenseDistanceManyToMany<float>(*query_tokenization_dist_, *float_queries,
                                 centers, distance_callback);

  for (DatapointIndex query_idx : IndicesOf(*float_queries)) {
    const auto max_centers = max_centers_override.empty()
                                 ? query_spilling_max_centers_
                                 : max_centers_override[query_idx];
    ConstSpan<float> distances = distance_mat[query_idx];
    results[query_idx].clear();
    const size_t nearest_center_index =
        std::distance(distances.begin(),
                      std::min_element(distances.begin(), distances.end()));
    const double nearest_center_distance = distances[nearest_center_index];

    double max_dist_to_consider;
    switch (query_spilling_type_) {
      case QuerySpillingConfig::NO_SPILLING:
        max_dist_to_consider = nearest_center_distance;
        break;
      case QuerySpillingConfig::MULTIPLICATIVE:
        max_dist_to_consider =
            nearest_center_distance * query_spilling_threshold_;
        break;
      case QuerySpillingConfig::ADDITIVE:
        max_dist_to_consider =
            nearest_center_distance + query_spilling_threshold_;
        break;
      case QuerySpillingConfig::ABSOLUTE_DISTANCE:
        max_dist_to_consider = query_spilling_threshold_;
        break;
      default:
        return InvalidArgumentError("Unknown spilling type.");
    }

    vector<pair<DatapointIndex, float>>& cur_res = results[query_idx];
    for (DatapointIndex center_idx : IndicesOf(distances)) {
      if (distances[center_idx] <= max_dist_to_consider) {
        cur_res.emplace_back(center_idx, distances[center_idx]);
      }
    }

    if (cur_res.size() > max_centers) {
      ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                                   max_centers - 1, cur_res.begin(),
                                   cur_res.end());
      cur_res.resize(max_centers);
    }

    ZipSortBranchOptimized(DistanceComparatorBranchOptimized(), cur_res.begin(),
                           cur_res.end());
  }

  return OkStatus();
}

template <typename T>
int32_t KMeansTreePartitioner<T>::n_tokens() const {
  DCHECK(kmeans_tree_)
      << "Can only query n_tokens after a KMeansTreePartitioner is built.";
  if (kmeans_tree_->is_flat()) {
    return LeafCenters().size();
  } else {
    return kmeans_tree_->n_tokens();
  }
}

template <typename T>
Normalization KMeansTreePartitioner<T>::NormalizationRequired() const {
  if (this->tokenization_mode() == UntypedPartitioner::QUERY) {
    return query_tokenization_dist_->NormalizationRequired();
  } else {
    DCHECK_EQ(this->tokenization_mode(), UntypedPartitioner::DATABASE);
    return database_tokenization_dist_->NormalizationRequired();
  }
}

namespace internal {
StatusOr<unique_ptr<SingleMachineSearcherBase<float>>>
CreateRecommendedAsymmetricSearcher(
    shared_ptr<DenseDataset<float>> dataset,
    shared_ptr<const DistanceMeasure> quantization_distance,
    int32_t num_neighbors, float epsilon = numeric_limits<float>::infinity(),
    bool with_exact_reordering = true, shared_ptr<ThreadPool> pool = nullptr,
    int num_clusters_per_block = 16, int num_dimension_per_block = 2);
}

template <typename T>
const SingleMachineSearcherBase<float>*
KMeansTreePartitioner<T>::TokenizationSearcher() const {
  if (this->tokenization_mode() == UntypedPartitioner::QUERY) {
    return query_tokenization_searcher_.get();
  } else {
    return database_tokenization_searcher_.get();
  }
}

template <typename T>
Status
KMeansTreePartitioner<T>::CreateAsymmetricHashingSearcherForQueryTokenization(
    bool with_exact_reordering) {
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Must train partitioner first before using searcher for "
        "tokenization.");
  }
  if (!kmeans_tree_->is_flat()) {
    return FailedPreconditionError(
        "Use searcher for tokenization only works for one_level_tree.");
  }
  if (!(query_spilling_type_ == QuerySpillingConfig::NO_SPILLING ||
        query_spilling_type_ == QuerySpillingConfig::ABSOLUTE_DISTANCE ||
        query_spilling_type_ == QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS)) {
    return FailedPreconditionError(
        "Searcher may be only used with NO_SPILLING, ABSOLUTE_DISTANCE "
        "spilling or FIXED_NUMBER_OF_CENTERS spilling.");
  }

  const auto& original_centers = kmeans_tree_->root()->Centers();
  auto centers = std::make_unique<DenseDataset<float>>();
  original_centers.ConvertType(centers.get());

  SCANN_ASSIGN_OR_RETURN(
      query_tokenization_searcher_,
      internal::CreateRecommendedAsymmetricSearcher(
          std::move(centers), query_tokenization_dist_,
          query_spilling_max_centers_, numeric_limits<float>::infinity(),
          with_exact_reordering));
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<
    T>::CreateAsymmetricHashingSearcherForDatabaseTokenization() {
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Must train partitioner first before using searcher for "
        "tokenization");
  }
  if (!kmeans_tree_->is_flat()) {
    return FailedPreconditionError(
        "Use searcher for tokenization only works for one_level_tree.");
  }
  if (kmeans_tree_->learned_spilling_type() !=
      DatabaseSpillingConfig::NO_SPILLING) {
    return FailedPreconditionError(
        "Searcher may be only used with NO_SPILLING spilling.");
  }

  const auto& original_centers = kmeans_tree_->root()->Centers();
  auto centers = std::make_unique<DenseDataset<float>>();
  original_centers.ConvertType(centers.get());

  SCANN_ASSIGN_OR_RETURN(database_tokenization_searcher_,
                         internal::CreateRecommendedAsymmetricSearcher(
                             std::move(centers), database_tokenization_dist_, 1,
                             numeric_limits<float>::infinity()));
  return OkStatus();
}

template <typename T>
StatusOr<vector<pair<DatapointIndex, float>>>
KMeansTreePartitioner<T>::TokenForDatapointBatchedImpl(
    const TypedDataset<T>& queries, ThreadPool* pool) const {
  const DenseDataset<T>& dense = *down_cast<const DenseDataset<T>*>(&queries);
  DenseDataset<float> float_query_storage;
  auto float_queries = ConvertToFloatIfNecessary(dense, &float_query_storage);

  const DenseDataset<float>& centers = kmeans_tree_->root()->Centers();
  if (centers.dimensionality() != queries.dimensionality()) {
    return FailedPreconditionError(
        "Incorrect query dimensionality.  Expected %d, got %d.\n",
        centers.dimensionality(), queries.dimensionality());
  }

  const auto& dist = (this->tokenization_mode() == UntypedPartitioner::QUERY)
                         ? *query_tokenization_dist_
                         : *database_tokenization_dist_;

  return DenseDistanceManyToManyTop1<float>(dist, *float_queries, centers,
                                            pool);
}

template <typename T>
Status KMeansTreePartitioner<T>::OrthogonalityAmplifiedTokenForDatapointBatched(
    const DenseDataset<T>& queries,
    ConstSpan<pair<DatapointIndex, float>> primary_centroids,
    MutableSpan<pair<DatapointIndex, float>> secondary_centroids,
    ThreadPool* pool) const {
  if (!kmeans_tree_->is_flat()) {
    return UnimplementedError(
        "Orthogonality amplification only works for one_level_tree.");
  }
  SCANN_RET_CHECK(queries.IsDense())
      << "Orthogonality amplification is only supported for dense data.";
  SCANN_RET_CHECK_EQ(primary_centroids.size(), secondary_centroids.size());
  SCANN_RET_CHECK_EQ(primary_centroids.size(), queries.size());
  if (primary_centroids.empty()) return OkStatus();

  const DenseDataset<float>& centers_dataset = LeafCenters();
  auto create_normalized_residual_dataset =
      [&](size_t start, size_t end) -> DenseDataset<float> {
    CHECK_GT(end, start);
    CHECK_LE(end, primary_centroids.size());
    DenseDataset<float> result;
    auto slice = primary_centroids.subspan(start, end - start);
    result.set_dimensionality(centers_dataset.dimensionality());
    result.Reserve(slice.size());
    vector<float> normalized_residual(result.dimensionality());
    for (size_t i : IndicesOf(slice)) {
      ComputeNormalizedResidual(queries[i + start],
                                centers_dataset[slice[i].first],
                                MakeMutableSpan(normalized_residual));
      result.AppendOrDie(MakeDatapointPtr(normalized_residual), "");
    }
    return result;
  };

  auto create_queries_dataset = [&](size_t start, size_t end) {
    CHECK_GT(end, start);
    CHECK_LE(end, primary_centroids.size());
    vector<float> result(queries.dimensionality() * (end - start));
    for (size_t i : Seq(start, end)) {
      auto dptr = queries[i];
      std::copy(dptr.values(), dptr.values() + dptr.dimensionality(),
                result.begin() + (i - start) * queries.dimensionality());
    }
    return DenseDataset<float>(std::move(result), end - start);
  };

  constexpr size_t kMaxBatchSize = 256;
  const size_t num_batches =
      DivRoundUp(primary_centroids.size(), kMaxBatchSize);
  return ParallelForWithStatus<1>(
      Seq(num_batches), pool, [&](size_t batch_idx) -> Status {
        const size_t start = batch_idx * kMaxBatchSize;
        const size_t end =
            std::min(start + kMaxBatchSize, primary_centroids.size());
        const size_t batch_size = end - start;
        SCANN_RET_CHECK_GE(batch_size, 1);
        MutableSpan<pair<DatapointIndex, float>> top1_span =
            secondary_centroids.subspan(start, batch_size);
        std::fill(top1_span.begin(), top1_span.end(),
                  std::make_pair(kInvalidDatapointIndex,
                                 numeric_limits<float>::infinity()));
        ManyToManyTop1Callback<float> top1_callback(top1_span);
        EpsilonFilteringCallback<float> eps_callback(top1_callback.epsilons(),
                                                     top1_callback);
        DenseManyToManyOrthogonalityAmplified(
            create_queries_dataset(start, end),
            create_normalized_residual_dataset(start, end),
            orthogonality_amplification_lambda_, LeafCenters(), nullptr,
            eps_callback);
        return OkStatus();
      });
}

SCANN_INSTANTIATE_TYPED_CLASS(, KMeansTreePartitioner);

}  // namespace research_scann
