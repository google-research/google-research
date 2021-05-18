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



#ifndef SCANN_BASE_SINGLE_MACHINE_BASE_H_
#define SCANN_BASE_SINGLE_MACHINE_BASE_H_

#include <cstdint>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/docid_collection.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/hashes/hashing_base.h"
#include "scann/metadata/metadata_getter.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/reordering_helper.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

template <typename T>
class SingleMachineSearcherBase;

class UntypedSingleMachineSearcherBase {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(UntypedSingleMachineSearcherBase);

  virtual ~UntypedSingleMachineSearcherBase();

  virtual research_scann::TypeTag TypeTag() const = 0;

  virtual const Dataset* dataset() const = 0;

  const DenseDataset<uint8_t>* hashed_dataset() const {
    return hashed_dataset_.get();
  }

  StatusOr<string_view> GetDocid(DatapointIndex i) const;

  Status set_docids(shared_ptr<const DocidCollectionInterface> docids);

  shared_ptr<const DocidCollectionInterface> docids() const { return docids_; }

  int64_t creation_timestamp() const { return creation_timestamp_; }
  void set_creation_timestamp(int64_t x) { creation_timestamp_ = x; }

  virtual bool needs_dataset() const { return true; }

  bool needs_hashed_dataset() const;

  virtual void ReleaseDataset() = 0;

  bool MaybeReleaseDataset() {
    if (!needs_dataset()) {
      ReleaseDataset();
      return true;
    }
    return false;
  }

  virtual void ReleaseHashedDataset() = 0;

  virtual void ReleaseDatasetAndDocids() = 0;

  void SetUnspecifiedParametersToDefaults(SearchParameters* params) const;

  Status SetMetadataGetter(shared_ptr<UntypedMetadataGetter> metadata_getter);

  virtual bool metadata_enabled() const = 0;

  int32_t default_pre_reordering_num_neighbors() const {
    return default_search_parameters_.pre_reordering_num_neighbors();
  }
  float default_pre_reordering_epsilon() const {
    return default_search_parameters_.pre_reordering_epsilon();
  }
  int32_t default_post_reordering_num_neighbors() const {
    return default_search_parameters_.post_reordering_num_neighbors();
  }
  float default_post_reordering_epsilon() const {
    return default_search_parameters_.post_reordering_epsilon();
  }

  Status EnableCrowding(
      std::vector<int64_t> datapoint_index_to_crowding_attribute);
  Status EnableCrowding(
      shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute);

  bool crowding_enabled() const {
    return datapoint_index_to_crowding_attribute_ != nullptr;
  }

  virtual bool supports_crowding() const { return false; }

  ConstSpan<int64_t> datapoint_index_to_crowding_attribute() const {
    ConstSpan<int64_t> result;
    if (datapoint_index_to_crowding_attribute_) {
      result = *datapoint_index_to_crowding_attribute_;
    }
    return result;
  }

  void DisableCrowding() {
    DisableCrowdingImpl();
    datapoint_index_to_crowding_attribute_ = nullptr;
  }

  StatusOr<DatapointIndex> DatasetSize() const;

  virtual int64_t num_active_dimensions() const {
    return (dataset() == nullptr) ? -1 : (dataset()->NumActiveDimensions());
  }

  virtual bool reordering_enabled() const = 0;

  virtual DatapointIndex optimal_batch_size() const;

  class PrecomputedMutationArtifacts : public VirtualDestructor {};

  virtual StatusOr<SingleMachineFactoryOptions>
  ExtractSingleMachineFactoryOptions() = 0;

 protected:
  virtual bool impl_needs_dataset() const;

  virtual bool impl_needs_hashed_dataset() const;

  virtual Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
    return OkStatus();
  }

  virtual void DisableCrowdingImpl() {}

 private:
  UntypedSingleMachineSearcherBase(
      const shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
      const int32_t default_pre_reordering_num_neighbors,
      const float default_pre_reordering_epsilon);

  UntypedSingleMachineSearcherBase() {}

  shared_ptr<const DenseDataset<uint8_t>> hashed_dataset_ = nullptr;

  shared_ptr<const DocidCollectionInterface> docids_;

  shared_ptr<UntypedMetadataGetter> metadata_getter_;

  SearchParameters default_search_parameters_;

  shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute_ = {};

  int64_t creation_timestamp_ = numeric_limits<int64_t>::min();

  bool mutator_outstanding_ = false;

  template <typename T>
  friend class SingleMachineSearcherBase;
};

template <typename T>
class SingleMachineSearcherBase : public UntypedSingleMachineSearcherBase {
 public:
  using DataType = T;

  SCANN_DECLARE_MOVE_ONLY_CLASS(SingleMachineSearcherBase);

  SingleMachineSearcherBase(
      shared_ptr<const TypedDataset<T>> dataset,
      shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
      int32_t default_pre_reordering_num_neighbors,
      float default_pre_reordering_epsilon);

  SingleMachineSearcherBase(shared_ptr<const TypedDataset<T>> dataset,
                            int32_t default_pre_reordering_num_neighbors,
                            float default_pre_reordering_epsilon);

  SingleMachineSearcherBase(int32_t default_pre_reordering_num_neighbors,
                            float default_pre_reordering_epsilon)
      : UntypedSingleMachineSearcherBase(nullptr,
                                         default_pre_reordering_num_neighbors,
                                         default_pre_reordering_epsilon) {}

  ~SingleMachineSearcherBase() override;

  const MetadataGetter<T>* metadata_getter() const {
    return down_cast<MetadataGetter<T>*>(metadata_getter_.get());
  }

  void set_metadata_getter(unique_ptr<MetadataGetter<T>> metadata_getter) {
    metadata_getter_ = std::move(metadata_getter);
  }

  bool metadata_enabled() const final { return metadata_getter_ != nullptr; }

  bool needs_dataset() const override;

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

  virtual Status FindNeighbors(const DatapointPtr<T>& query,
                               const SearchParameters& params,
                               NNResultsVector* result) const;

  Status FindNeighbors(const DatapointPtr<T>& query,
                       NNResultsVector* result) const {
    return FindNeighbors(query, default_search_parameters_, result);
  }

  Status FindNeighborsNoSortNoExactReorder(const DatapointPtr<T>& query,
                                           const SearchParameters& params,
                                           NNResultsVector* result) const;

  Status FindNeighborsBatched(const TypedDataset<T>& queries,
                              MutableSpan<NNResultsVector> results) const;
  Status FindNeighborsBatched(const TypedDataset<T>& queries,
                              ConstSpan<SearchParameters> params,
                              MutableSpan<NNResultsVector> results) const;
  Status FindNeighborsBatchedNoSortNoExactReorder(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  virtual Status PreprocessQueryIntoParamsUnlocked(
      const DatapointPtr<T>& query, SearchParameters& search_params) const {
    search_params.set_unlocked_query_preprocessing_results(nullptr);
    return OkStatus();
  }

  Status GetNeighborProto(const pair<DatapointIndex, float> neighbor,
                          const DatapointPtr<T>& query,
                          NearestNeighbors::Neighbor* result) const;

  Status GetNeighborProtoNoMetadata(const pair<DatapointIndex, float> neighbor,
                                    const DatapointPtr<T>& query,
                                    NearestNeighbors::Neighbor* result) const;

  research_scann::TypeTag TypeTag() const final { return TagForType<T>(); }

  const TypedDataset<T>* dataset() const final { return dataset_.get(); }

  shared_ptr<const TypedDataset<T>> shared_dataset() const { return dataset_; }

  void ReleaseDataset() final;
  void ReleaseHashedDataset() final;
  void ReleaseDatasetAndDocids() final;

  DatapointPtr<T> GetDatapointPtr(DatapointIndex i) const {
    DCHECK(dataset_);
    return (*dataset_)[i];
  }

  void EnableReordering(shared_ptr<const ReorderingInterface<T>> reorder_helper,
                        const int32_t default_post_reordering_num_neighbors,
                        const float default_post_reordering_epsilon) {
    reordering_helper_ = reorder_helper;
    default_search_parameters_.set_post_reordering_num_neighbors(
        default_post_reordering_num_neighbors);
    default_search_parameters_.set_post_reordering_epsilon(
        default_post_reordering_epsilon);
  }

  void DisableReordering() { reordering_helper_ = nullptr; }

  void EnableExactReordering(
      shared_ptr<const DistanceMeasure> exact_reordering_distance,
      const int32_t default_post_reordering_num_neighbors,
      const float default_post_reordering_epsilon) {
    EnableReordering(std::make_shared<ExactReorderingHelper<T>>(
                         exact_reordering_distance, dataset_),
                     default_post_reordering_num_neighbors,
                     default_post_reordering_epsilon);
  }

  bool reordering_enabled() const final {
    return reordering_helper_ != nullptr;
  }

  void DisableExactReordering() { DisableReordering(); }

  bool exact_reordering_enabled() const {
    return (reordering_helper_ &&
            reordering_helper_->name() == "ExactReordering");
  }

  bool fixed_point_reordering_enabled() const;

  const ReorderingInterface<T>& reordering_helper() const {
    return *reordering_helper_;
  }

 protected:
  SingleMachineSearcherBase() {}

  Status BaseInitFromDatasetAndConfig(
      shared_ptr<const TypedDataset<T>> dataset,
      shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
      const ScannConfig& config);

  virtual Status FindNeighborsImpl(const DatapointPtr<T>& query,
                                   const SearchParameters& params,
                                   NNResultsVector* result) const = 0;

  virtual Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

 private:
  Status PopulateDefaultParameters(const ScannConfig& config);
  Status BaseInitImpl();

  Status ReorderResults(const DatapointPtr<T>& query,
                        const SearchParameters& params,
                        NNResultsVector* result) const;

  Status SortAndDropResults(NNResultsVector* result,
                            const SearchParameters& params) const;

  shared_ptr<const TypedDataset<T>> dataset_ = nullptr;

  shared_ptr<const ReorderingInterface<T>> reordering_helper_ = nullptr;

  friend class Mutator;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, SingleMachineSearcherBase);

}  // namespace research_scann

#endif
