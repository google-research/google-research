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



#include "scann/base/single_machine_base.h"

#include <typeinfo>

#include "absl/flags/flag.h"
#include "scann/base/search_parameters.h"
#include "tensorflow/core/platform/cpu_info.h"

#include "absl/strings/match.h"
#include "scann/utils/factory_helpers.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace scann_ops {

UntypedSingleMachineSearcherBase::~UntypedSingleMachineSearcherBase() {}

StatusOr<string_view> UntypedSingleMachineSearcherBase::GetDocid(
    DatapointIndex i) const {
  if (!docids_) {
    return FailedPreconditionError(
        "This SingleMachineSearcherBase instance does not have access to "
        "docids.  Did you call ReleaseDatasetAndDocids?");
  }

  const size_t n_docids = docids_->size();
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
    shared_ptr<vector<int64_t>> datapoint_index_to_crowding_attribute) {
  if (!supports_crowding()) {
    return UnimplementedError("Crowding not supported for this searcher.");
  }
  if (crowding_enabled_) {
    return FailedPreconditionError("Crowding already enabled.");
  }
  SCANN_RETURN_IF_ERROR(
      EnableCrowdingImpl(*datapoint_index_to_crowding_attribute));
  datapoint_index_to_crowding_attribute_ =
      std::move(datapoint_index_to_crowding_attribute);
  crowding_enabled_ = true;
  return OkStatus();
}

StatusOr<DatapointIndex> UntypedSingleMachineSearcherBase::DatasetSize() const {
  if (dataset()) {
    return dataset()->size();
  } else if (compressed_dataset()) {
    return compressed_dataset()->size();
  } else if (hashed_dataset()) {
    return hashed_dataset()->size();
  } else if (docids_) {
    return docids_->size();
  } else {
    return FailedPreconditionError(
        "Dataset size is not known for this searcher.");
  }
}

StatusOr<SingleMachineFactoryOptions>
UntypedSingleMachineSearcherBase::ExtractSingleMachineFactoryOptions() {
  SingleMachineFactoryOptions opts;

  opts.compressed_dataset =
      std::const_pointer_cast<DenseDataset<uint8_t>>(compressed_dataset_);
  opts.hashed_dataset =
      std::const_pointer_cast<DenseDataset<uint8_t>>(hashed_dataset_);

  opts.crowding_attributes = std::const_pointer_cast<vector<int64_t>>(
      datapoint_index_to_crowding_attribute_);
  opts.creation_timestamp = creation_timestamp_;
  return opts;
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
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon)
    : hashed_dataset_(hashed_dataset),
      default_search_parameters_(default_pre_reordering_num_neighbors,
                                 default_pre_reordering_epsilon,
                                 default_pre_reordering_num_neighbors,
                                 default_pre_reordering_epsilon) {
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
  TF_CHECK_OK(BaseInitImpl());
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

bool UntypedSingleMachineSearcherBase::needs_hashed_dataset() const {
  return impl_needs_hashed_dataset() ||

         (hashed_dataset_ && mutator_outstanding_);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighbors(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  DCHECK_LE((compressed_reordering_enabled() + exact_reordering_enabled()), 1);
  SCANN_RETURN_IF_ERROR(
      FindNeighborsNoSortNoExactReorder(query, params, result));

  if (reordering_helper_) {
    SCANN_RETURN_IF_ERROR(ReorderResults(query, params, result));
  }

  return SortAndDropResults(result, params);
}

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsNoSortNoExactReorder(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  bool reordering_enabled =
      compressed_reordering_enabled() || exact_reordering_enabled();
  SCANN_RETURN_IF_ERROR(params.Validate(reordering_enabled));
  if (!this->supports_crowding() && params.pre_reordering_crowding_enabled()) {
    return InvalidArgumentError(
        std::string(
            "Crowding is enabled but not supported for searchers of type ") +
        typeid(*this).name() + ".");
  }

  if (dataset() && !dataset()->empty() &&
      query.dimensionality() != dataset()->dimensionality()) {
    return FailedPreconditionError(
        StrFormat("Query dimensionality (%u) does not match database "
                  "dimensionality (%u)",
                  static_cast<unsigned long long_t>(query.dimensionality()),
                  static_cast<unsigned long long_t>(dataset()->dimensionality())));
  }

  return FindNeighborsImpl(query, params, result);
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
  DCHECK_LE((compressed_reordering_enabled() + exact_reordering_enabled()), 1);
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

template <typename T>
Status SingleMachineSearcherBase<T>::FindNeighborsBatchedNoSortNoExactReorder(
    const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  if (queries.size() != params.size()) {
    return InvalidArgumentError(
        "queries.size != params.size in FindNeighbors batched (%d vs. %d).",
        queries.size(), params.size());
  }

  if (queries.size() != results.size()) {
    return InvalidArgumentError(
        "queries.size != results.size in FindNeighbors batched (%d vs. %d).",
        queries.size(), results.size());
  }

  bool reordering_enabled =
      compressed_reordering_enabled() || exact_reordering_enabled();
  for (const SearchParameters& p : params) {
    SCANN_RETURN_IF_ERROR(p.Validate(reordering_enabled));
  }

  if (dataset() && !dataset()->empty() &&
      queries.dimensionality() != dataset()->dimensionality()) {
    return FailedPreconditionError(
        "Query dimensionality (%u) does not match database dimensionality (%u)",
        queries.dimensionality(), dataset()->dimensionality());
  }

  return FindNeighborsBatchedImpl(queries, params, results);
}

template <typename T>
Status SingleMachineSearcherBase<T>::GetNeighborProto(
    const pair<DatapointIndex, float> neighbor, const DatapointPtr<T>& query,
    NearestNeighbors::Neighbor* result) const {
  SCANN_RETURN_IF_ERROR(GetNeighborProtoNoMetadata(neighbor, query, result));
  if (!metadata_enabled()) return OkStatus();

  Status status = metadata_getter()->GetMetadata(
      dataset(), query, neighbor.first, result->mutable_metadata());
  if (!status.ok()) result->Clear();
  return status;
}

template <typename T>
Status SingleMachineSearcherBase<T>::GetNeighborProtoNoMetadata(
    const pair<DatapointIndex, float> neighbor, const DatapointPtr<T>& query,
    NearestNeighbors::Neighbor* result) const {
  DCHECK(result);
  result->Clear();
  TF_ASSIGN_OR_RETURN(auto docid, GetDocid(neighbor.first));
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

  if (!dataset() && compressed_dataset_) {
    DCHECK_EQ(docids_.get(), hashed_dataset_->docids().get());
    docids_ = compressed_dataset_->docids();
  }

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
Status SingleMachineSearcherBase<T>::ReorderResults(
    const DatapointPtr<T>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  if (params.post_reordering_num_neighbors() == 1) {
    TF_ASSIGN_OR_RETURN(
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

template <typename T>
Status SingleMachineSearcherBase<T>::SortAndDropResults(
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

namespace {

template <typename T>
bool SameDocidsInstance(
    const shared_ptr<const DocidCollectionInterface>& docids,
    const TypedDataset<T>* dataset) {
  if (!dataset) return false;
  return docids == dataset->docids();
}

}  // namespace

template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::PrepareForBaseMutation(
    SingleMachineSearcherBase<T>* searcher) {
  searcher_ = searcher;
  searcher_->mutator_outstanding_ = true;
  if (searcher->dataset_) {
    TF_ASSIGN_OR_RETURN(dataset_mutator_, searcher->dataset_->GetMutator());
  }
  if (searcher->hashed_dataset_) {
    TF_ASSIGN_OR_RETURN(hashed_dataset_mutator_,
                        searcher->hashed_dataset_->GetMutator());
  }
  if (searcher_->reordering_helper_ &&
      searcher_->reordering_helper_->owns_mutation_data_structures()) {
    TF_ASSIGN_OR_RETURN(reordering_mutator_,
                        searcher->reordering_helper_->GetMutator());
  }
  if (searcher->docids_ &&
      !SameDocidsInstance(searcher->docids_, searcher->dataset_.get()) &&
      !SameDocidsInstance(searcher->docids_, searcher->hashed_dataset_.get())) {
    TF_ASSIGN_OR_RETURN(docid_mutator_, searcher->docids_->GetMutator());
  }
  return OkStatus();
}

template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::GetNextDatapointIndex() const {
  DatapointIndex result = kInvalidDatapointIndex;
  if (searcher_->dataset_) {
    result = searcher_->dataset_->size();

    if (searcher_->docids_)
      SCANN_RET_CHECK_EQ(result, searcher_->docids_->size());
    if (searcher_->hashed_dataset_) {
      SCANN_RET_CHECK_EQ(result, searcher_->hashed_dataset_->size());
    }
  } else if (searcher_->hashed_dataset_) {
    result = searcher_->hashed_dataset()->size();
    if (searcher_->docids_)
      SCANN_RET_CHECK_EQ(result, searcher_->docids_->size());
  } else if (searcher_->docids_) {
    result = searcher_->docids_->size();
  }
  return result;
}

template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::AddDatapointToBase(
    const DatapointPtr<T>& dptr, const DatapointPtr<uint8_t>& hashed,
    string_view docid) {
  TF_ASSIGN_OR_RETURN(const DatapointIndex result, GetNextDatapointIndex());
  if (dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->AddDatapoint(dptr, docid));
  }
  if (hashed_dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(hashed_dataset_mutator_->AddDatapoint(hashed, docid));
  }
  if (docid_mutator_) {
    SCANN_RETURN_IF_ERROR(docid_mutator_->AddDatapoint(docid));
  }
  if (reordering_mutator_) {
    TF_ASSIGN_OR_RETURN(auto idx, reordering_mutator_->AddDatapoint(dptr));
    SCANN_RET_CHECK_EQ(result, idx);
  }
  return result;
}

template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::UpdateDatapointInBase(
    const DatapointPtr<T>& dptr, const DatapointPtr<uint8_t>& hashed,
    DatapointIndex idx) {
  if (dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->UpdateDatapoint(dptr, idx));
  }
  if (hashed_dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(
        hashed_dataset_mutator_->UpdateDatapoint(hashed, idx));
  }
  if (reordering_mutator_) {
    SCANN_RETURN_IF_ERROR(reordering_mutator_->UpdateDatapoint(dptr, idx));
  }
  return OkStatus();
}

template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::RemoveDatapointFromBase(
    DatapointIndex idx) {
  SCANN_RETURN_IF_ERROR(GetNextDatapointIndex().status());

  DatapointIndex result = kInvalidDatapointIndex;
  if (dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->RemoveDatapoint(idx));
    result = searcher_->dataset_->size();
  }
  if (hashed_dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(hashed_dataset_mutator_->RemoveDatapoint(idx));
    result = searcher_->hashed_dataset_->size();
  }
  if (docid_mutator_) {
    SCANN_RETURN_IF_ERROR(docid_mutator_->RemoveDatapoint(idx));
    result = searcher_->docids_->size();
  }
  if (reordering_mutator_) {
    TF_ASSIGN_OR_RETURN(auto swapped_from,
                        reordering_mutator_->RemoveDatapoint(idx));
    if (result != kInvalidDatapointIndex) {
      SCANN_RET_CHECK_EQ(swapped_from, result);
    }
  }
  return result;
}

template <typename T>
void SingleMachineSearcherBase<T>::Mutator::ReserveInBase(
    DatapointIndex num_datapoints) {
  if (dataset_mutator_) dataset_mutator_->Reserve(num_datapoints);
  if (hashed_dataset_mutator_) hashed_dataset_mutator_->Reserve(num_datapoints);
  if (reordering_mutator_) reordering_mutator_->Reserve(num_datapoints);
  if (docid_mutator_) docid_mutator_->Reserve(num_datapoints);
}

template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::AddDatapointWithMetadata(
    const DatapointPtr<T>& dptr, const GenericFeatureVector& gfv,
    MutationMetadata* md) {
  if (searcher_->metadata_enabled()) {
    SCANN_RETURN_IF_ERROR(searcher_->metadata_getter_->AppendMetadata(gfv));
  }
  return AddDatapoint(dptr, gfv.data_id_str(), md);
}

template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::UpdateDatapointWithMetadata(
    const DatapointPtr<T>& dptr, const GenericFeatureVector& gfv,
    DatapointIndex idx, MutationMetadata* md) {
  if (searcher_->metadata_enabled()) {
    SCANN_RETURN_IF_ERROR(
        searcher_->metadata_getter_->UpdateMetadata(idx, gfv));
  }
  TF_ASSIGN_OR_RETURN(const DatapointIndex new_idx,
                      UpdateDatapoint(dptr, idx, md));
  SCANN_RET_CHECK_EQ(idx, new_idx)
      << "Datapoint indices should not change when "
         "updating a datapoint in place.";
  return idx;
}

template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::RemoveDatapointWithMetadata(
    DatapointIndex idx) {
  if (searcher_->metadata_enabled()) {
    SCANN_RETURN_IF_ERROR(searcher_->metadata_getter_->RemoveMetadata(idx));
  }
  return RemoveDatapoint(idx);
}

template <typename T>
bool SingleMachineSearcherBase<T>::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  if (dataset_mutator_) {
    return dataset_mutator_->LookupDatapointIndex(docid, index);
  }
  if (hashed_dataset_mutator_) {
    return hashed_dataset_mutator_->LookupDatapointIndex(docid, index);
  }
  if (docid_mutator_) return docid_mutator_->LookupDatapointIndex(docid, index);
  return false;
}

template <typename T>
bool SingleMachineSearcherBase<T>::fixed_point_reordering_enabled() const {
  return (reordering_helper_ &&
          absl::StartsWith(reordering_helper_->name(), "FixedPoint"));
}

SCANN_INSTANTIATE_TYPED_CLASS(, SingleMachineSearcherBase);

}  // namespace scann_ops
}  // namespace tensorflow
