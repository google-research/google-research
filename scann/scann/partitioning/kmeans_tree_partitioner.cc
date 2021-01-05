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

#include "scann/partitioning/kmeans_tree_partitioner.h"

#include "absl/base/internal/spinlock.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/partitioning/kmeans_tree_partitioner.pb.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "scann/utils/zip_sort.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace scann_ops {

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
      query_tokenization_dist_(query_tokenization_dist) {
  SetIsOneLevelTree();
}

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
  SetIsOneLevelTree();
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
  result->query_tokenization_searcher_ = query_tokenization_searcher_;
  result->populate_residual_stdev_ = populate_residual_stdev_;
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
  SetIsOneLevelTree();
  return OkStatus();
}

template <typename T>
void KMeansTreePartitioner<T>::set_query_spilling_threshold(double val) {
  query_spilling_threshold_ = val;
}

constexpr int kAhMultiplierSpilling = 10;

constexpr int kAhMultiplierNoSpilling = 100;

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, KMeansTreeSearchResult* result) const {
  DCHECK(result);
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Cannot query a KMeansTreePartitioner before training.");
  }

  if (this->tokenization_mode() == UntypedPartitioner::QUERY) {
    if (query_tokenization_type_ == ASYMMETRIC_HASHING) {
      int pre_reordering_num_neighbors =
          TokenizationSearcher()->reordering_enabled() ? kAhMultiplierNoSpilling
                                                       : 1;
      return TokenForDatapointUseSearcher(dptr, result,
                                          pre_reordering_num_neighbors);
    } else {
      vector<KMeansTreeSearchResult> result_vec;
      SCANN_RETURN_IF_ERROR(
          kmeans_tree_->Tokenize(dptr, *query_tokenization_dist_,
                                 KMeansTree::TokenizationOptions::NoSpilling(
                                     static_cast<KMeansTree::TokenizationType>(
                                         query_tokenization_type_),
                                     populate_residual_stdev_),
                                 &result_vec));
      *result = result_vec[0];
      return OkStatus();
    }
  } else if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    if (database_tokenization_type_ == ASYMMETRIC_HASHING) {
      int pre_reordering_num_neighbors =
          TokenizationSearcher()->reordering_enabled() ? kAhMultiplierNoSpilling
                                                       : 1;
      return TokenForDatapointUseSearcher(dptr, result,
                                          pre_reordering_num_neighbors);
    } else {
      vector<KMeansTreeSearchResult> result_vec;
      SCANN_RETURN_IF_ERROR(
          kmeans_tree_->Tokenize(dptr, *database_tokenization_dist_,
                                 KMeansTree::TokenizationOptions::NoSpilling(
                                     static_cast<KMeansTree::TokenizationType>(
                                         database_tokenization_type_),
                                     populate_residual_stdev_),
                                 &result_vec));
      *result = result_vec[0];
      return OkStatus();
    }
  }
  return InternalError(
      absl::StrCat("Invalid tokenization mode:  ", this->tokenization_mode()));
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapointBatched(
    const TypedDataset<T>& queries, vector<int32_t>* results,
    thread::ThreadPool* pool) const {
  if (query_tokenization_type_ != FLOAT || queries.IsSparse() ||
      !is_one_level_tree_) {
    return Partitioner<T>::TokenForDatapointBatched(queries, results);
  }

  const DenseDataset<T>& dense = *down_cast<const DenseDataset<T>*>(&queries);
  DenseDataset<float> float_query_storage;
  auto float_queries = ConvertToFloatIfNecessary(dense, &float_query_storage);

  const DenseDataset<float>& centers = kmeans_tree_->root()->FloatCenters();
  if (centers.dimensionality() != queries.dimensionality()) {
    return FailedPreconditionError(
        "Incorrect query dimensionality.  Expected %d, got %d.\n",
        centers.dimensionality(), queries.dimensionality());
  }

  const auto& dist = (this->tokenization_mode() == UntypedPartitioner::QUERY)
                         ? *query_tokenization_dist_
                         : *database_tokenization_dist_;

  vector<pair<uint32_t, float>> top1_results =
      DenseDistanceManyToManyTop1<float>(dist, *float_queries, centers, pool);

  results->resize(queries.size());
  for (size_t j : Seq(queries.size())) {
    (*results)[j] = static_cast<int32_t>(top1_results[j].first);
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<KMeansTreeSearchResult>* result) const {
  DCHECK(result);

  if (this->tokenization_mode() == UntypedPartitioner::QUERY) {
    const auto max_centers = max_centers_override > 0
                                 ? max_centers_override
                                 : query_spilling_max_centers_;

    if (query_tokenization_type_ == ASYMMETRIC_HASHING) {
      int pre_reordering_num_neighbors =
          TokenizationSearcher()->reordering_enabled()
              ? max_centers * kAhMultiplierSpilling
              : max_centers;
      return TokensForDatapointWithSpillingUseSearcher(
          dptr, result, max_centers, pre_reordering_num_neighbors);
    }

    return kmeans_tree_->Tokenize(
        dptr, *query_tokenization_dist_,
        KMeansTree::TokenizationOptions::UserSpecifiedSpilling(
            query_spilling_type_, query_spilling_threshold_, max_centers,
            static_cast<KMeansTree::TokenizationType>(query_tokenization_type_),
            populate_residual_stdev_),
        result);
  } else if (this->tokenization_mode() == UntypedPartitioner::DATABASE) {
    if (database_spilling_fixed_number_of_centers_ > 0) {
      if (database_tokenization_type_ == ASYMMETRIC_HASHING) {
        int pre_reordering_num_neighbors =
            TokenizationSearcher()->reordering_enabled()
                ? database_spilling_fixed_number_of_centers_ *
                      kAhMultiplierSpilling
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
                    query_tokenization_type_),
                populate_residual_stdev_),
            result);
      }
    }

    if (database_tokenization_type_ == ASYMMETRIC_HASHING) {
      if (kmeans_tree_->learned_spilling_type() ==
          DatabaseSpillingConfig::NO_SPILLING) {
        result->resize(1);
        return TokenForDatapoint(dptr, &result->front());

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
                database_tokenization_type_),
            populate_residual_stdev_),
        result);
  } else {
    return InternalError(absl::StrCat("Unknown tokenization mode:  ",
                                      this->tokenization_mode()));
  }
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapointUseSearcher(
    const DatapointPtr<T>& dptr, KMeansTreeSearchResult* result,
    int32_t pre_reordering_num_neighbors) const {
  if (!TokenizationSearcher()) {
    return FailedPreconditionError(
        "CreateAsymmetricHashingSearcherForTokenization must "
        "be called first.");
  }
  Datapoint<float> dp;
  DatapointPtr<float> query = ToFloat(dptr, &dp);

  SearchParameters params(pre_reordering_num_neighbors,
                          numeric_limits<float>::infinity(), 1,
                          numeric_limits<float>::infinity());

  NNResultsVector search_result;
  Status status =
      TokenizationSearcher()->FindNeighbors(query, params, &search_result);
  if (!status.ok()) return status;

  DCHECK_LE(search_result[0].first, kint32max);

  DCHECK(is_one_level_tree_);
  const auto* root = kmeans_tree_->root();
  const auto token = search_result[0].first;
  result->node = &root->Children()[token];
  DCHECK_EQ(result->node->LeafId(), token);
  result->distance_to_center = search_result[0].second;
  result->residual_stdev =
      populate_residual_stdev_ && token < root->residual_stdevs().size()
          ? root->residual_stdevs()[token]
          : 1.0;
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokenForDatapoint(const DatapointPtr<T>& dptr,
                                                   int32_t* result) const {
  KMeansTreeSearchResult tree_res;
  SCANN_RETURN_IF_ERROR(TokenForDatapoint(dptr, &tree_res));
  *result = tree_res.node->LeafId();
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingAndOverride(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<int32_t>* result) const {
  vector<KMeansTreeSearchResult> result_raw;
  SCANN_RETURN_IF_ERROR(
      TokensForDatapointWithSpilling(dptr, max_centers_override, &result_raw));
  result->clear();
  result->reserve(result_raw.size());
  for (auto& elem : result_raw) {
    result->push_back(elem.node->LeafId());
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingUseSearcher(
    const DatapointPtr<T>& dptr, vector<KMeansTreeSearchResult>* result,
    int32_t num_neighbors, int32_t pre_reordering_num_neighbors) const {
  if (!TokenizationSearcher()) {
    return FailedPreconditionError(
        "CreateAsymmetricHashingSearcherForTokenization must "
        "be called first.");
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
  NNResultsVector search_result;
  Status status =
      TokenizationSearcher()->FindNeighbors(query, params, &search_result);

  if (!status.ok()) return status;

  DCHECK(is_one_level_tree_);
  result->clear();
  result->reserve(search_result.size());
  const auto* root = kmeans_tree_->root();
  for (const auto& elem : search_result) {
    DCHECK_LE(elem.first, kint32max);
    result->emplace_back(KMeansTreeSearchResult{
        &root->Children()[elem.first], elem.second,
        populate_residual_stdev_ && elem.first < root->residual_stdevs().size()
            ? root->residual_stdevs()[elem.first]
            : 1.0});
    DCHECK_EQ(elem.first, result->back().node->LeafId());
  }
  return OkStatus();
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
    const DatapointPtr<T>& dptr, int32_t token,
    bool normalize_residual_by_cluster_stdev) const {
  DatapointPtr<float> center = kmeans_tree()->CenterForToken(token);
  if (normalize_residual_by_cluster_stdev) {
    if (!populate_residual_stdev_) {
      return InvalidArgumentError(
          "normalize_residual_by_cluster_stdev can only apply on partitioner "
          "with populate_residual_stdev enabled");
    }
    TF_ASSIGN_OR_RETURN(float multiplier,
                        kmeans_tree()->ResidualStdevForToken(token));
    float inv_multiplier = 1.0 / multiplier;
    return ResidualizeImpl<float>(dptr, center, inv_multiplier);
  }
  return ResidualizeImpl<float>(dptr, center);
}

template <typename T>
const DenseDataset<float>& KMeansTreePartitioner<T>::LeafCenters() const {
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
KMeansTreePartitioner<T>::TokenizeDatabase(
    const TypedDataset<T>& database, thread::ThreadPool* pool_or_null) const {
  if (this->tokenization_mode() != UntypedPartitioner::DATABASE) {
    return FailedPreconditionError(
        "Cannot run TokenizeDatabase when not in database tokenization mode.");
  }
  if (typeid(*database_tokenization_dist_) == typeid(const SquaredL2Distance) &&
      is_one_level_tree_ && database.IsDense() &&
      kmeans_tree_->learned_spilling_type() ==
          DatabaseSpillingConfig::NO_SPILLING &&
      (IsSame<T, float>() || IsSame<T, double>()) &&
      (database_tokenization_type_ == FLOAT)) {
    const DenseDataset<T>& dense_database =
        *down_cast<const DenseDataset<T>*>(&database);
    TF_ASSIGN_OR_RETURN(
        auto datapoint_index_to_result,
        TokenizeDatabaseImplFastPath(dense_database, pool_or_null));
    vector<std::vector<DatapointIndex>> token_to_datapoint_index(
        this->n_tokens());
    for (DatapointIndex dp_index = 0;
         dp_index < datapoint_index_to_result.size(); ++dp_index) {
      const int32_t token = datapoint_index_to_result[dp_index].node->LeafId();
      token_to_datapoint_index[token].push_back(dp_index);
    }
    for (auto& elem : token_to_datapoint_index) {
      elem.shrink_to_fit();
    }
    return std::move(token_to_datapoint_index);
  }

  return Partitioner<T>::TokenizeDatabase(database, pool_or_null);
}

template <typename T>
StatusOr<vector<KMeansTreeSearchResult>>
KMeansTreePartitioner<T>::TokenizeDatabaseImplFastPath(
    const DenseDataset<T>& database, thread::ThreadPool* pool_or_null) const {
  vector<KMeansTreeSearchResult> datapoint_index_to_result;
  if (kmeans_tree_->root()->IsLeaf()) {
    datapoint_index_to_result.resize(
        database.size(), KMeansTreeSearchResult{kmeans_tree_->root(), NAN});
    return std::move(datapoint_index_to_result);
  }

  if (database_tokenization_type_ == FLOAT) {
    DCHECK_EQ(database_tokenization_type_, FLOAT);
    TF_ASSIGN_OR_RETURN(
        datapoint_index_to_result,
        TokenizeDatabaseImplFastPath<float>(
            database, kmeans_tree_->root()->FloatCenters(), pool_or_null));
  }
  return std::move(datapoint_index_to_result);
}

template <typename T>
template <typename CenterType>
enable_if_t<IsSame<T, CenterType>(), StatusOr<vector<KMeansTreeSearchResult>>>
KMeansTreePartitioner<T>::TokenizeDatabaseImplFastPath(
    const DenseDataset<T>& database, const DenseDataset<CenterType>& centers,
    thread::ThreadPool* pool_or_null) const {
  SquaredL2Distance dist;
  auto nearest_centers =
      DenseDistanceManyToManyTop1<T>(dist, database, centers, pool_or_null);
  return PostprocessNearestCenters<CenterType>(nearest_centers);
}

template <typename T>
template <typename CenterType>
enable_if_t<!IsSame<T, CenterType>(), StatusOr<vector<KMeansTreeSearchResult>>>
KMeansTreePartitioner<T>::TokenizeDatabaseImplFastPath(
    const DenseDataset<T>& database, const DenseDataset<CenterType>& centers,
    thread::ThreadPool* pool_or_null) const {
  constexpr size_t kBatchSize = 128;
  vector<pair<DatapointIndex, CenterType>> nearest_centers(database.size());
  SquaredL2Distance dist;

  ParallelFor<1>(
      SeqWithStride<kBatchSize>(0, database.size()), pool_or_null,
      [&](size_t batch_begin) {
        const size_t batch_end =
            std::min<size_t>(batch_begin + kBatchSize, database.size());
        DenseDataset<CenterType> batch_submatrix =
            GetBatchSubmatrix<CenterType>(database, batch_begin, batch_end);
        auto local_results = DenseDistanceManyToManyTop1<CenterType>(
            dist, batch_submatrix, centers);
        std::copy(local_results.begin(), local_results.end(),
                  nearest_centers.begin() + batch_begin);
      });

  return PostprocessNearestCenters<CenterType>(nearest_centers);
}

template <typename T>
template <typename FloatT>
vector<KMeansTreeSearchResult>
KMeansTreePartitioner<T>::PostprocessNearestCenters(
    ConstSpan<pair<DatapointIndex, FloatT>> nearest_centers) const {
  vector<KMeansTreeSearchResult> datapoint_index_to_result(
      nearest_centers.size());
  const auto* root = kmeans_tree_->root();
  const ConstSpan<KMeansTreeNode> nodes_array = root->Children();
  for (DatapointIndex dp_idx : IndicesOf(nearest_centers)) {
    DCHECK_LT(nearest_centers[dp_idx].first,
              numeric_limits<DatapointIndex>::max());
    const auto token = nearest_centers[dp_idx].first;
    datapoint_index_to_result[dp_idx] = {
        &nodes_array[token], nearest_centers[dp_idx].second,
        populate_residual_stdev_ && token < root->residual_stdevs().size()
            ? root->residual_stdevs()[token]
            : 1.0};
  }

  return datapoint_index_to_result;
}

template <typename T>
Status
KMeansTreePartitioner<T>::TokensForDatapointWithSpillingBatchedAndOverride(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<vector<int32_t>> results) const {
  vector<vector<KMeansTreeSearchResult>> raw_results(queries.size());
  SCANN_RETURN_IF_ERROR(TokensForDatapointWithSpillingBatched(
      queries, max_centers_override, MakeMutableSpan(raw_results)));
  for (size_t i = 0; i < results.size(); ++i) {
    vector<int32_t>& cur_results = results[i];
    auto& cur_raw_results = raw_results[i];
    cur_results.clear();
    cur_results.reserve(cur_raw_results.size());
    for (auto& elem : cur_raw_results) {
      cur_results.push_back(elem.node->LeafId());
    }
  }
  return OkStatus();
}

template <typename T>
Status KMeansTreePartitioner<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<vector<KMeansTreeSearchResult>> results) const {
  if (!max_centers_override.empty() &&
      queries.size() != max_centers_override.size())
    return InvalidArgumentError(
        "The max_centers override must have the same "
        "size as batched queries.");

  if (this->tokenization_mode() != UntypedPartitioner::QUERY ||
      !SupportsLowLevelQueryBatching() || !queries.IsDense()) {
    for (DatapointIndex i = 0; i < queries.size(); ++i) {
      const auto max_centers =
          max_centers_override.empty() ? 0 : max_centers_override[i];
      SCANN_RETURN_IF_ERROR(
          TokensForDatapointWithSpilling(queries[i], max_centers, &results[i]));
    }
    return OkStatus();
  }

  const DenseDataset<float>& centers = kmeans_tree_->root()->FloatCenters();
  if (centers.dimensionality() != queries.dimensionality()) {
    return FailedPreconditionError(
        "Incorrect query dimensionality.  Expected %d, got %d.\n",
        centers.dimensionality(), queries.dimensionality());
  }

  DenseDataset<float> float_query_storage;
  auto float_queries = ConvertToFloatIfNecessary(
      *down_cast<const DenseDataset<T>*>(&queries), &float_query_storage);

  auto to_kmeans_tree_search_results =
      [&](ConstSpan<pair<DatapointIndex, float>> child_centers)
      -> vector<KMeansTreeSearchResult> {
    const auto* root = kmeans_tree_->root();
    vector<KMeansTreeSearchResult> this_query_results;
    this_query_results.reserve(child_centers.size());
    for (const auto& child_center : child_centers) {
      const auto token = child_center.first;
      const KMeansTreeNode* node = &root->Children()[token];
      DCHECK_EQ(child_center.first, node->LeafId());
      KMeansTreeSearchResult result;
      result.node = node;
      result.distance_to_center = child_center.second;
      result.residual_stdev =
          populate_residual_stdev_ && token < root->residual_stdevs().size()
              ? root->residual_stdevs()[token]
              : 1.0;
      this_query_results.push_back(result);
    }
    return this_query_results;
  };

  if (query_spilling_type_ == QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    vector<FastTopNeighbors<float>> ftns(float_queries->size());
    for (DatapointIndex query_idx : IndicesOf(*float_queries)) {
      const auto max_centers = max_centers_override.empty()
                                   ? query_spilling_max_centers_
                                   : max_centers_override[query_idx];
      ftns[query_idx] = FastTopNeighbors<float>(max_centers);
    }
    DenseDistanceManyToMany<float>(
        *query_tokenization_dist_, *float_queries, centers,
        [&ftns](MutableSpan<float> dists, DatapointIndex base_dp_idx,
                DatapointIndex query_idx) {
          FastTopNeighbors<float>::Mutator mut;
          ftns[query_idx].AcquireMutator(&mut);
          mut.PushDistanceBlock(dists, base_dp_idx);
        });
    NNResultsVector child_centers;
    for (DatapointIndex query_idx : IndicesOf(*float_queries)) {
      ftns[query_idx].FinishUnsorted(&child_centers);
      ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                                   ftns[query_idx].max_results() - 1,
                                   child_centers.begin(), child_centers.end());
      results[query_idx] = to_kmeans_tree_search_results(child_centers);
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

    NNResultsVector child_centers;
    for (DatapointIndex center_idx : IndicesOf(distances)) {
      if (distances[center_idx] <= max_dist_to_consider) {
        child_centers.emplace_back(center_idx, distances[center_idx]);
      }
    }

    if (child_centers.size() > max_centers) {
      ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                                   max_centers - 1, child_centers.begin(),
                                   child_centers.end());
      child_centers.resize(max_centers);
    }

    ZipSortBranchOptimized(DistanceComparatorBranchOptimized(),
                           child_centers.begin(), child_centers.end());
    results[query_idx] = to_kmeans_tree_search_results(child_centers);
  }

  return OkStatus();
}

template <typename T>
int32_t KMeansTreePartitioner<T>::n_tokens() const {
  DCHECK(kmeans_tree_)
      << "Can only query n_tokens after a KMeansTreePartitioner is built.";
  return kmeans_tree_->n_tokens();
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

template <typename T>
void KMeansTreePartitioner<T>::SetIsOneLevelTree() {
  const KMeansTreeNode* root = kmeans_tree_->root();
  is_one_level_tree_ = true;
  for (const KMeansTreeNode& child : root->Children()) {
    if (!child.IsLeaf()) {
      is_one_level_tree_ = false;
      return;
    }
  }
}

namespace internal {
StatusOr<unique_ptr<SingleMachineSearcherBase<float>>>
CreateRecommendedAsymmetricSearcher(
    shared_ptr<DenseDataset<float>> dataset,
    shared_ptr<const DistanceMeasure> quantization_distance,
    int32_t num_neighbors, float epsilon = numeric_limits<float>::infinity(),
    bool with_exact_reordering = true,
    shared_ptr<thread::ThreadPool> pool = nullptr,
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
  if (!is_one_level_tree_) {
    return FailedPreconditionError(
        "Use searcher for tokenization only works for one_level_tree.");
  }
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Must train partitioner first before using searcher for "
        "tokenization.");
  }
  if (!(query_spilling_type_ == QuerySpillingConfig::NO_SPILLING ||
        query_spilling_type_ == QuerySpillingConfig::ABSOLUTE_DISTANCE ||
        query_spilling_type_ == QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS)) {
    return FailedPreconditionError(
        "Searcher may be only used with NO_SPILLING, ABSOLUTE_DISTANCE "
        "spilling or FIXED_NUMBER_OF_CENTERS spilling.");
  }

  const auto& original_centers = kmeans_tree_->root()->Centers();
  auto centers = unique_ptr<DenseDataset<float>>(new DenseDataset<float>);
  original_centers.ConvertType(centers.get());

  TF_ASSIGN_OR_RETURN(
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
  if (!is_one_level_tree_) {
    return FailedPreconditionError(
        "Use searcher for tokenization only works for one_level_tree.");
  }
  if (!kmeans_tree_) {
    return FailedPreconditionError(
        "Must train partitioner first before using searcher for "
        "tokenization");
  }
  if (kmeans_tree_->learned_spilling_type() !=
      DatabaseSpillingConfig::NO_SPILLING) {
    return FailedPreconditionError(
        "Searcher may be only used with NO_SPILLING spilling.");
  }

  const auto& original_centers = kmeans_tree_->root()->Centers();
  auto centers = absl::make_unique<DenseDataset<float>>();
  original_centers.ConvertType(centers.get());

  TF_ASSIGN_OR_RETURN(database_tokenization_searcher_,
                      internal::CreateRecommendedAsymmetricSearcher(
                          std::move(centers), database_tokenization_dist_, 1,
                          numeric_limits<float>::infinity()));
  return OkStatus();
}

SCANN_INSTANTIATE_TYPED_CLASS(, KMeansTreePartitioner);

}  // namespace scann_ops
}  // namespace tensorflow
