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



#include "scann/base/single_machine_base.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/prefetch.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/brute_force/brute_force.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/distance_measures/one_to_many/one_to_many_symmetric.h"
#include "scann/metadata/metadata_getter.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/proto/results.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/multi_stage_batch_pipeline.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

UntypedSingleMachineSearcherBase::~UntypedSingleMachineSearcherBase() {}

StatusOr<string_view> UntypedSingleMachineSearcherBase::GetDocid(
    DatapointIndex i) const {
  if (!docids_) {
    return FailedPreconditionError(
        "This SingleMachineSearcherBase instance does not have access to "
        "docids.  Did you call ReleaseDatasetAndDocids?");
  }

  const size_t n_docids = docids_->size();
  const Dataset* dataset = this->dataset();
  if (dataset) {
    SCANN_RET_CHECK_EQ(n_docids, dataset->size())
        << "Dataset size and docids size have diverged.  (Datapoint index "
           "requested to GetDocid = "
        << i << ")  This likely indicates an internal error in ScaNN.";
  }
  if (i >= n_docids) {
    return InvalidArgumentError("Datapoint index (%d) is >= dataset size (%d).",
                                i, n_docids);
  }

  return docids_->Get(i);
}

Status UntypedSingleMachineSearcherBase::set_docids(
    shared_ptr<const DocidCollectionInterface> docids) {
  if (dataset() || hashed_dataset()) {
    return FailedPreconditionError(
        "UntypedSingleMachineSearcherBase::set_docids may only be called "
        "on instances constructed using the constructor that does not accept "
        "a Dataset.");
  }

  if (docids_) {
    return FailedPreconditionError(
        "UntypedSingleMachineSearcherBase::set_docids may not be called if "
        "the docid array is not empty.  This can happen if set_docids has "
        "already been called on this instance, or if this instance was "
        "constructed using the constructor that takes a Dataset and then "
        "ReleaseDataset was called.");
  }

  docids_ = std::move(docids);
  return OkStatus();
}

void UntypedSingleMachineSearcherBase::SetUnspecifiedParametersToDefaults(
    SearchParameters* params) const {
  params->SetUnspecifiedParametersFrom(default_search_parameters_);
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    vector<int64_t> datapoint_index_to_crowding_attribute) {
  return EnableCrowding(std::make_shared<vector<int64_t>>(
      std::move(datapoint_index_to_crowding_attribute)));
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    vector<int64_t> datapoint_index_to_crowding_attribute,
    vector<std::string> crowding_dimension_names) {
  return EnableCrowding(std::make_shared<vector<int64_t>>(
                            std::move(datapoint_index_to_crowding_attribute)),
                        std::make_shared<vector<std::string>>(
                            std::move(crowding_dimension_names)));
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    shared_ptr<vector<int64_t>> datapoint_index_to_crowding_attribute) {
  return EnableCrowding(datapoint_index_to_crowding_attribute,
                        std::make_shared<vector<std::string>>());
}

Status UntypedSingleMachineSearcherBase::EnableCrowding(
    shared_ptr<vector<int64_t>> datapoint_index_to_crowding_attribute,
    shared_ptr<vector<std::string>> crowding_dimension_names) {
  SCANN_RET_CHECK(datapoint_index_to_crowding_attribute);
  if (!supports_crowding()) {
    return UnimplementedError("Crowding not supported for this searcher.");
  }
  if (crowding_enabled()) {
    return FailedPreconditionError("Crowding already enabled.");
  }
  SCANN_RETURN_IF_ERROR(
      EnableCrowdingImpl(*datapoint_index_to_crowding_attribute,
                         crowding_dimension_names != nullptr
                             ? absl::MakeConstSpan(*crowding_dimension_names)
                             : absl::Span<std::string>()));
  datapoint_index_to_crowding_attribute_ =
      std::move(datapoint_index_to_crowding_attribute);
  crowding_dimension_names_ = std::move(crowding_dimension_names);
  return OkStatus();
}

StatusOr<DatapointIndex> UntypedSingleMachineSearcherBase::DatasetSize() const {
  if (dataset()) {
    return dataset()->size();
  } else if (hashed_dataset()) {
    return hashed_dataset()->size();
  } else if (docids_) {
    return docids_->size();
  } else {
    Status status =
        FailedPreconditionError("Dataset size is not known for this searcher.");
    return status;
  }
}

vector<uint32_t> UntypedSingleMachineSearcherBase::SizeByPartition() const {
  return {};
}

bool UntypedSingleMachineSearcherBase::impl_needs_dataset() const {
  return true;
}

bool UntypedSingleMachineSearcherBase::impl_needs_hashed_dataset() const {
  return true;
}

DatapointIndex UntypedSingleMachineSearcherBase::optimal_batch_size() const {
  return 1;
}

UntypedSingleMachineSearcherBase::UntypedSingleMachineSearcherBase(
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : default_search_parameters_(
          default_pre_reordering_num_neighbors, default_pre_reordering_epsilon,
          default_pre_reordering_num_neighbors, default_pre_reordering_epsilon),
      hashed_dataset_(hashed_dataset) {
  if (default_pre_reordering_num_neighbors <= 0) {
    LOG(FATAL) << "default_pre_reordering_num_neighbors must be > 0, not "
               << default_pre_reordering_num_neighbors << ".";
  }

  if (std::isnan(default_pre_reordering_epsilon)) {
    LOG(FATAL) << "default_pre_reordering_epsilon must be non-NaN.";
  }
}

template <typename T>
SingleMachineSearcherBase<T>::SingleMachineSearcherBase(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon)
    : UntypedSingleMachineSearcherBase(hashed_dataset,
                                       default_pre_reordering_num_neighbors,
                                       default_pre_reordering_epsilon),
      dataset_(dataset) {
  CHECK_OK(BaseInitImpl());
}

template <typename T>
Status SingleMachineSearcherBase<T>::BaseInitImpl() {
  if (hashed_dataset_ && dataset_ &&
      dataset_->size() != hashed_dataset_->size()) {
    return FailedPreconditionError(
        "If both dataset and hashed_dataset are provided, they must have the "
        "same size.");
  }

  if (dataset_) {
    docids_ = dataset_->docids();
  } else if (hashed_dataset_) {
    docids_ = hashed_dataset_->docids();
  } else {
    DCHECK(!docids_);
  }
  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::BaseInitFromDatasetAndConfig(
    shared_ptr<const TypedDataset<T>> dataset,
    shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
    const ScannConfig& config) {
  dataset_ = std::move(dataset);
  hashed_dataset_ = std::move(hashed_dataset);
  SCANN_RETURN_IF_ERROR(PopulateDefaultParameters(config));
  return BaseInitImpl();
}

template <typename T>
Status SingleMachineSearcherBase<T>::PopulateDefaultParameters(
    const ScannConfig& config) {
  GenericSearchParameters params;
  SCANN_RETURN_IF_ERROR(params.PopulateValuesFromScannConfig(config));
  const bool params_has_pre_norm =
      params.pre_reordering_dist->NormalizationRequired() != NONE;
  const bool params_has_exact_norm =
      params.reordering_dist->NormalizationRequired() != NONE;
  const bool dataset_has_norm =
      dataset_ && dataset_->normalization() !=
                      params.pre_reordering_dist->NormalizationRequired();
  if (params_has_pre_norm && !dataset_has_norm) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the pre-reordering distance "
        "measure.");
  }
  if (params_has_exact_norm && !dataset_has_norm) {
    return InvalidArgumentError(
        "Dataset not correctly normalized for the exact distance measure.");
  }
  const int32_t k = params.pre_reordering_num_neighbors;
  const float epsilon = params.pre_reordering_epsilon;
  default_search_parameters_ = SearchParameters(k, epsilon, k, epsilon);
  return OkStatus();
}

template <typename T>
SingleMachineSearcherBase<T>::SingleMachineSearcherBase(
    shared_ptr<const TypedDataset<T>> dataset,
    int32_t default_pre_reordering_num_neighbors,
    float default_pre_reordering_epsilon)
    : SingleMachineSearcherBase(dataset, nullptr,
                                default_pre_reordering_num_neighbors,
                                default_pre_reordering_epsilon) {}

template <typename T>
SingleMachineSearcherBase<T>::~SingleMachineSearcherBase() {}

Status UntypedSingleMachineSearcherBase::SetMetadataGetter(
    shared_ptr<UntypedMetadataGetter> metadata_getter) {
  if (metadata_getter && metadata_getter->TypeTag() != this->TypeTag()) {
    return FailedPreconditionError(
        "SetMetadataGetter called with a MetadataGetter<%s>. Expected "
        "MetadataGetter<%s>.",
        TypeNameFromTag(metadata_getter->TypeTag()),
        TypeNameFromTag(this->TypeTag()));
  }
  metadata_getter_ = std::move(metadata_getter);
  return OkStatus();
}

template <typename T>
bool SingleMachineSearcherBase<T>::needs_dataset() const {
  return impl_needs_dataset() ||
         (reordering_enabled() && reordering_helper_->needs_dataset()) ||
         (metadata_enabled() && metadata_getter_->needs_dataset()) ||

         (dataset_ && mutator_outstanding_);
}

template <typename T>
StatusOr<SingleMachineFactoryOptions>
SingleMachineSearcherBase<T>::ExtractSingleMachineFactoryOptions() {
  SingleMachineFactoryOptions opts;

  opts.hashed_dataset =
      std::const_pointer_cast<DenseDataset<uint8_t>>(hashed_dataset_);

  opts.crowding_attributes = std::const_pointer_cast<vector<int64_t>>(
      datapoint_index_to_crowding_attribute_);
  if (reordering_helper_)
    reordering_helper_->AppendDataToSingleMachineFactoryOptions(&opts);
  return opts;
}

template <typename T>
StatusOr<shared_ptr<const DenseDataset<float>>>
SingleMachineSearcherBase<T>::SharedFloatDatasetIfNeeded() {
  if (!needs_dataset()) return shared_ptr<const DenseDataset<float>>(nullptr);
  if (dataset() == nullptr)
    return InternalError(
        "Searcher needs original dataset but none is present.");
  auto dataset =
      std::dynamic_pointer_cast<const DenseDataset<float>>(shared_dataset());
  if (dataset == nullptr)
    return InternalError("Failed to cast to DenseDataset<float>.");
  return dataset;
}

template <typename T>
StatusOr<shared_ptr<const DenseDataset<float>>>
SingleMachineSearcherBase<T>::ReconstructFloatDataset() {
  SCANN_ASSIGN_OR_RETURN(auto dataset, SharedFloatDatasetIfNeeded());
  if (dataset != nullptr) {
    return dataset;
  } else if (reordering_enabled()) {
    return reordering_helper().ReconstructFloatDataset();
  }
  return shared_ptr<const DenseDataset<float>>(nullptr);
}

bool UntypedSingleMachineSearcherBase::needs_hashed_dataset() const {
  return impl_needs_hashed_dataset() ||

         (hashed_dataset_ && mutator_outstanding_);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighbors(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  SCANN_RET_CHECK(query.IsFinite())
      << "Cannot query ScaNN with vectors that contain NaNs or infinity.";
  DCHECK(result);
  SCANN_RETURN_IF_ERROR(
      FindNeighborsNoSortNoExactReorder(query, params, result));

  if (reordering_helper_) {
    SCANN_RETURN_IF_ERROR(ReorderResults(query, params, result));
  }

  SCANN_RETURN_IF_ERROR(SortAndDropResults(result, params));

  if (params.num_random_neighbors()) {
    SCANN_RETURN_IF_ERROR(SampleRandomNeighbors(query, params, result));
  }

  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsNoSortNoExactReorder(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  bool reordering_enabled = exact_reordering_enabled();
  SCANN_RETURN_IF_ERROR(params.Validate(reordering_enabled));
  if (!this->supports_crowding() && params.pre_reordering_crowding_enabled()) {
    return InvalidArgumentError(
        std::string(
            "Crowding is enabled but not supported for searchers of type ") +
        typeid(*this).name() + ".");
  }
  if (!this->crowding_dimension_names().empty() &&
      params.pre_reordering_crowding_enabled()) {
    return InvalidArgumentError(
        std::string(
            "Received request with pre-reordering crowding enabled, but "
            "multi-dimensional crowding is not supported for searchers of "
            "type ") +
        typeid(*this).name() + ".");
  }
  if (!this->crowding_enabled() && params.crowding_enabled()) {
    return InvalidArgumentError(
        "Crowding is enabled for query but not enabled in searcher.");
  }

  std::optional<DimensionIndex> db_dim;
  if (dataset() && !dataset()->empty()) {
    db_dim = dataset()->dimensionality();
  } else if (reordering_helper_) {
    auto dataset = reordering_helper_->dataset();
    if (dataset && !dataset->empty()) db_dim = dataset->dimensionality();
  }
  if (db_dim && *db_dim != query.dimensionality()) {
    return FailedPreconditionError(
        StrFormat("Query dimensionality (%d) does not match database "
                  "dimensionality (%d)",
                  query.dimensionality(), *db_dim));
  }
  if (params.restrict_whitelist() && docids_ &&
      params.restrict_whitelist()->size() > docids_->size()) {
    const std::string dataset_size =
        (dataset()) ? StrCat(dataset()->size()) : "unknown";
    return OutOfRangeError(
        "The number of datapoints in the restrict allowlist (%d) is greater "
        "than the number of docids in the dataset (%d).  Dataset object size = "
        "%s.",
        params.restrict_whitelist()->size(), docids_->size(), dataset_size);
  }

  return FindNeighborsImpl(query, params, result);
}

template <typename T>
Status SingleMachineSearcherBase<T>::SampleRandomNeighbors(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  vector<pair<DatapointIndex, float>> sampled;
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    FastTopNeighbors<float> top_n(params.num_random_neighbors());
    SCANN_RETURN_IF_ERROR(SampleRandomNeighborsImpl(params, &top_n));
    top_n.FinishUnsorted(&sampled);
  }
  SCANN_RETURN_IF_ERROR(PropagateDistances(query, params, &sampled));
  result->insert(result->end(), sampled.begin(), sampled.end());
  return OkStatus();
}

template <typename T>
template <typename TopN>
Status SingleMachineSearcherBase<T>::SampleRandomNeighborsImpl(
    const SearchParameters& params, TopN* top_n_ptr) const {
  typename TopN::Mutator mutator;
  top_n_ptr->AcquireMutator(&mutator);
  float min_keep_distance = mutator.epsilon();

  absl::BitGen rng;
  auto push = [&](DatapointIndex dp_idx) {
    float distance = absl::Uniform<float>(rng, 0.0, 1.0);
    if (distance <= min_keep_distance && mutator.Push(dp_idx, distance)) {
      mutator.GarbageCollect();
      min_keep_distance = mutator.epsilon();
    }
  };

  if (params.restricts_enabled()) {
    return UnimplementedError("Restricts not supported.");
  } else {
    SCANN_ASSIGN_OR_RETURN(DatapointIndex dataset_size, DatasetSize());
    for (DatapointIndex idx = 0; idx < dataset_size; ++idx) {
      push(idx);
    }
  }
  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::PropagateDistances(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  if (!config_.has_value()) {
    return FailedPreconditionError(
        "Config is not set, can not determine distance measure");
  }
  if (dataset_ != nullptr) {
    SCANN_ASSIGN_OR_RETURN(auto dist,
                           GetDistanceMeasure(config_->distance_measure()));
    for (auto& elem : *result) {
      elem.second = dist->GetDistance(query, GetDatapointPtr(elem.first));
    }
    return OkStatus();
  }
  if (reordering_helper_ != nullptr) {
    return reordering_helper_->ComputeDistancesForReordering(query, result);
  }
  return FailedPreconditionError(
      "Cannot propagate distances without a dataset and without reordering.");
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatched(
    const TypedDataset<T>& queries, MutableSpan<NNResultsVector> result) const {
  vector<SearchParameters> params(queries.size());
  for (auto& p : params) {
    p.SetUnspecifiedParametersFrom(default_search_parameters_);
  }
  return FindNeighborsBatched(queries, params, result);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatched(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  SCANN_RETURN_IF_ERROR(
      FindNeighborsBatchedNoSortNoExactReorder(queries, params, results));

  if (reordering_helper_) {
    for (DatapointIndex i = 0; i < queries.size(); ++i) {
      SCANN_RETURN_IF_ERROR(ReorderResults(queries[i], params[i], &results[i]));
    }
  }

  for (DatapointIndex i = 0; i < results.size(); ++i) {
    SCANN_RETURN_IF_ERROR(SortAndDropResults(&results[i], params[i]));
  }

  return OkStatus();
}

template <typename ResultElem>
Status UntypedSingleMachineSearcherBase::ValidateFindNeighborsBatched(
    const Dataset& queries, ConstSpan<SearchParameters> params,
    MutableSpan<ResultElem> results) const {
  if (!params.empty() ||
      !std::is_same_v<ResultElem, FastTopNeighbors<float>*>) {
    if (queries.size() != params.size()) {
      return InvalidArgumentError(
          "queries.size != params.size in FindNeighbors batched (%d vs. %d).",
          queries.size(), params.size());
    }
  }

  if (queries.size() != results.size()) {
    return InvalidArgumentError(
        "queries.size != results.size in FindNeighbors batched (%d vs. %d).",
        queries.size(), results.size());
  }
  for (auto [query_idx, param] : Enumerate(params)) {
    if (!this->supports_crowding() && param.pre_reordering_crowding_enabled()) {
      return InvalidArgumentError(
          absl::Substitute("Crowding is enabled for query (index $0) but not "
                           "supported for searchers of type $1.",
                           query_idx, typeid(*this).name()));
    }
    if (!this->crowding_dimension_names().empty() &&
        param.pre_reordering_crowding_enabled()) {
      return InvalidArgumentError(absl::Substitute(
          "Received request with pre-reordering crowding enabled, but "
          "multi-dimensional crowding is not supported for searchers of "
          "type $0.",
          typeid(*this).name()));
    }
    if (!this->crowding_enabled() && param.crowding_enabled()) {
      return InvalidArgumentError(
          absl::Substitute("Crowding is enabled for query (index $0) but not "
                           "enabled in searcher.",
                           query_idx));
    }
    if (param.restrict_whitelist() && docids_ &&
        param.restrict_whitelist()->size() > docids_->size()) {
      const std::string dataset_size =
          (dataset()) ? StrCat(dataset()->size()) : "unknown";
      return OutOfRangeError(
          "The number of datapoints in the restrict allowlist (%d) is greater "
          "than the number of docids in the dataset (%d).  Dataset object size "
          "= %s.",
          param.restrict_whitelist()->size(), docids_->size(), dataset_size);
    }
  }

  bool reordering_enabled = exact_reordering_enabled();
  for (const SearchParameters& p : params) {
    SCANN_RETURN_IF_ERROR(p.Validate(reordering_enabled));
  }

  if (dataset() && !dataset()->empty() &&
      queries.dimensionality() != dataset()->dimensionality()) {
    return FailedPreconditionError(
        "Query dimensionality (%u) does not match database dimensionality (%u)",
        queries.dimensionality(), dataset()->dimensionality());
  }
  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedNoSortNoExactReorder(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  SCANN_RETURN_IF_ERROR(ValidateFindNeighborsBatched(queries, params, results));
  return FindNeighborsBatchedImpl(queries, params, results);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedNoSortNoExactReorder(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<FastTopNeighbors<float>*> results,
    ConstSpan<DatapointIndex> datapoint_index_lookup) const {
  SCANN_RETURN_IF_ERROR(ValidateFindNeighborsBatched(queries, params, results));
  return FindNeighborsBatchedImpl(queries, params, results,
                                  datapoint_index_lookup);
}

template <typename T>
Status SingleMachineSearcherBase<T>::GetNeighborProto(
    pair<DatapointIndex, float> neighbor, const DatapointPtr<T>& query,
    NearestNeighbors::Neighbor* result) const {
  SCANN_RETURN_IF_ERROR(GetNeighborProtoNoMetadata(neighbor, result));

  if (!metadata_enabled()) return OkStatus();

  Status status = metadata_getter()->GetMetadata(
      dataset(), query, neighbor.first, result->mutable_metadata());
  if (!status.ok()) result->Clear();
  return status;
}

Status UntypedSingleMachineSearcherBase::GetNeighborProtoNoMetadata(
    pair<DatapointIndex, float> neighbor,
    NearestNeighbors::Neighbor* result) const {
  DCHECK(result);
  result->Clear();
  SCANN_ASSIGN_OR_RETURN(auto docid, GetDocid(neighbor.first));
  result->set_docid(std::string(docid));
  result->set_distance(neighbor.second);
  if (crowding_enabled()) {
    result->set_crowding_attribute(
        (*datapoint_index_to_crowding_attribute_)[neighbor.first]);
  }
  return OkStatus();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseDataset() {
  if (needs_dataset()) {
    LOG(FATAL) << "Cannot release dataset for this instance.";
    return;
  }

  if (!dataset_) return;

  if (hashed_dataset()) {
    DCHECK_EQ(docids_.get(), dataset_->docids().get());
    docids_ = hashed_dataset_->docids();
  }

  dataset_.reset();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseHashedDataset() {
  if (!hashed_dataset_) return;
  hashed_dataset_.reset();
}

template <typename T>
void SingleMachineSearcherBase<T>::ReleaseDatasetAndDocids() {
  if (needs_dataset()) {
    LOG(FATAL) << "Cannot release dataset for this instance.";
    return;
  }

  dataset_.reset();
  docids_.reset();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  DCHECK_EQ(queries.size(), params.size());
  DCHECK_EQ(queries.size(), results.size());
  for (DatapointIndex i = 0; i < queries.size(); ++i) {
    SCANN_RETURN_IF_ERROR(
        FindNeighborsImpl(queries[i], params[i], &results[i]));
  }

  return OkStatus();
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedImpl(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<FastTopNeighbors<float>*> results,
    ConstSpan<DatapointIndex> datapoint_index_mapping) const {
  if (!params.empty()) SCANN_RET_CHECK_EQ(queries.size(), params.size());
  SCANN_RET_CHECK_EQ(queries.size(), results.size());
  vector<NNResultsVector> vec_results(queries.size());
  vector<SearchParameters> default_params_storage;
  if (params.empty()) {
    default_params_storage.resize(queries.size());
    params = default_params_storage;
    for (size_t i : IndicesOf(results)) {
      const size_t max_results = results[i]->max_results();
      SCANN_RET_CHECK_GT(max_results, 0);
      default_params_storage[i].set_pre_reordering_epsilon(
          results[i]->epsilon());
      default_params_storage[i].set_pre_reordering_num_neighbors(max_results);
    }
  }
  SCANN_RETURN_IF_ERROR(
      FindNeighborsBatchedImpl(queries, params, MakeMutableSpan(vec_results)));

  auto do_push = [&](auto map_datapoint_index) SCANN_INLINE_LAMBDA {
    for (DatapointIndex q_idx : IndicesOf(vec_results)) {
      DCHECK(!params[q_idx].pre_reordering_crowding_enabled());
      DCHECK(results[q_idx]);
      FastTopNeighbors<float>::Mutator mut;
      results[q_idx]->AcquireMutator(&mut);
      float eps =
          std::min(params[q_idx].pre_reordering_epsilon(), mut.epsilon());
      for (const auto& elem : vec_results[q_idx]) {
        if (elem.second <= eps &&
            mut.Push(map_datapoint_index(elem.first), elem.second)) {
          mut.GarbageCollect();
          eps = mut.epsilon();
        }
      }
    }
  };
  if (datapoint_index_mapping.empty()) {
    do_push([](DatapointIndex dp_idx) { return dp_idx; });
  } else {
    do_push([&datapoint_index_mapping](DatapointIndex dp_idx) {
      DCHECK_LT(dp_idx, datapoint_index_mapping.size());
      return datapoint_index_mapping[dp_idx];
    });
  }
  return OkStatus();
}

template <typename T>
StatusOr<const SingleMachineSearcherBase<T>*>
SingleMachineSearcherBase<T>::CreateBruteForceSearcher(
    const DistanceMeasureConfig& distance_config,
    unique_ptr<SingleMachineSearcherBase<T>>* storage) const {
  SCANN_RET_CHECK(storage);
  SingleMachineSearcherBase<T>* result = nullptr;
  if (dataset()) {
    SCANN_ASSIGN_OR_RETURN(auto dist, GetDistanceMeasure(distance_config));
    SCANN_RET_CHECK(storage);
    storage->reset(new BruteForceSearcher<T>(
        std::move(dist), dataset_, default_post_reordering_num_neighbors(),
        default_post_reordering_epsilon()));
    result = storage->get();
  } else if (reordering_helper_) {
    SCANN_RET_CHECK(storage);
    SCANN_ASSIGN_OR_RETURN(*storage,
                           reordering_helper_->CreateBruteForceSearcher(
                               default_post_reordering_num_neighbors(),
                               default_post_reordering_epsilon()));
    result = storage->get();
    return result;
  }

  if (!result) {
    return FailedPreconditionError(
        "Cannot create brute force searcher from a non-brute force searcher "
        "without reordering enabled.");
  }
  result->docids_ = docids_;
  result->metadata_getter_ = metadata_getter_;
  result->datapoint_index_to_crowding_attribute_ =
      datapoint_index_to_crowding_attribute_;
  result->creation_timestamp_ = creation_timestamp_;
  return result;
}

template <typename T>
Status SingleMachineSearcherBase<T>::ReorderResults(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  if (params.post_reordering_num_neighbors() == 1) {
    SCANN_ASSIGN_OR_RETURN(
        auto top1,
        reordering_helper_->ComputeTop1ReorderingDistance(query, result));
    if (!result->empty() && top1.second < params.post_reordering_epsilon() &&
        top1.first != kInvalidDatapointIndex) {
      result->resize(1);
      result->at(0) = top1;
    } else {
      result->resize(0);
    }
  } else {
    SCANN_RETURN_IF_ERROR(
        reordering_helper_->ComputeDistancesForReordering(query, result));
  }
  return OkStatus();
}

Status UntypedSingleMachineSearcherBase::SortAndDropResults(
    NNResultsVector* result, const SearchParameters& params) const {
  if (reordering_enabled()) {
    if (params.post_reordering_num_neighbors() == 1) {
      return OkStatus();
    }

    if (params.post_reordering_epsilon() < numeric_limits<float>::infinity()) {
      auto it = std::partition(
          result->begin(), result->end(),
          [&params](const pair<DatapointIndex, float>& arg) {
            return arg.second <= params.post_reordering_epsilon();
          });
      const size_t new_size = it - result->begin();
      result->resize(new_size);
    }

    if (params.post_reordering_crowding_enabled()) {
      return FailedPreconditionError("Crowding is not supported.");
    } else {
      RemoveNeighborsPastLimit(params.post_reordering_num_neighbors(), result);
    }
  }

  if (params.sort_results()) {
    ZipSortBranchOptimized(DistanceComparatorBranchOptimized(), result->begin(),
                           result->end());
  }
  return OkStatus();
}

template <typename T>
bool SingleMachineSearcherBase<T>::fixed_point_reordering_enabled() const {
  return (reordering_helper_ &&
          absl::StartsWith(reordering_helper_->name(), "FixedPoint"));
}

SCANN_INSTANTIATE_TYPED_CLASS(, SingleMachineSearcherBase);

}  // namespace research_scann
