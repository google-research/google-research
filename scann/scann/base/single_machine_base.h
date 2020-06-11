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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__BASE_SINGLE_MACHINE_BASE_H_
#define SCANN__BASE_SINGLE_MACHINE_BASE_H_

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

namespace tensorflow {
namespace scann_ops {

template <typename T>
class SingleMachineSearcherBase;

class UntypedSingleMachineSearcherBase {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(UntypedSingleMachineSearcherBase);

  virtual ~UntypedSingleMachineSearcherBase();

  virtual tensorflow::scann_ops::TypeTag TypeTag() const = 0;

  virtual const Dataset* dataset() const = 0;

  const DenseDataset<uint8_t>* hashed_dataset() const {
    return hashed_dataset_.get();
  }

  shared_ptr<const DenseDataset<uint8_t>> shared_hashed_dataset() const {
    return hashed_dataset_;
  }

  const DenseDataset<uint8_t>* compressed_dataset() const {
    return compressed_dataset_.get();
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

  bool crowding_enabled() const { return crowding_enabled_; }

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
    crowding_enabled_ = false;
  }

  StatusOr<DatapointIndex> DatasetSize() const;

  virtual int64_t num_active_dimensions() const {
    return (dataset() == nullptr) ? -1 : (dataset()->NumActiveDimensions());
  }

  virtual bool reordering_enabled() const = 0;

  virtual DatapointIndex optimal_batch_size() const;

  class MutationMetadata : public VirtualDestructor {};

  class UntypedMutator {
   public:
    virtual ~UntypedMutator() {}

    virtual Status RemoveDatapoint(string_view docid) = 0;

    virtual bool LookupDatapointIndex(string_view docid,
                                      DatapointIndex* index) const = 0;

    virtual void Reserve(size_t size) = 0;

    virtual Status RemoveDatapoint(DatapointIndex index) = 0;

    virtual Status RemoveDatapointWithMetadata(DatapointIndex idx) = 0;

    using DatapointIndexRenameFn =
        std::function<void(DatapointIndex old_idx, DatapointIndex new_idx)>;
    void AddOnDatapointIndexRenameFn(DatapointIndexRenameFn fn) {
      on_datapoint_index_rename_fns_.push_back(fn);
    }

   protected:
    void OnDatapointIndexRename(DatapointIndex old_idx,
                                DatapointIndex new_idx) const {
      for (auto& fn : on_datapoint_index_rename_fns_) {
        fn(old_idx, new_idx);
      }
    }

    virtual StatusOr<DatapointIndex> RemoveDatapointFromBase(
        DatapointIndex index) = 0;

    virtual void ReserveInBase(DatapointIndex num_datapoints) = 0;

   private:
    vector<DatapointIndexRenameFn> on_datapoint_index_rename_fns_;
  };

  virtual StatusOr<typename UntypedSingleMachineSearcherBase::UntypedMutator*>
  GetUntypedMutator() const = 0;

  virtual StatusOr<SingleMachineFactoryOptions>
  ExtractSingleMachineFactoryOptions();

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

  shared_ptr<const DenseDataset<uint8_t>> compressed_dataset_ = nullptr;

  shared_ptr<const DocidCollectionInterface> docids_;

  shared_ptr<UntypedMetadataGetter> metadata_getter_;

  SearchParameters default_search_parameters_;

  shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute_ = {};

  int64_t creation_timestamp_ = numeric_limits<int64_t>::min();

  bool crowding_enabled_ = false;

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

  virtual StatusOr<
      unique_ptr<SearchParameters::UnlockedQueryPreprocessingResults>>
  UnlockedPreprocessQuery(const DatapointPtr<T>& query) const {
    return {nullptr};
  }

  Status GetNeighborProto(const pair<DatapointIndex, float> neighbor,
                          const DatapointPtr<T>& query,
                          NearestNeighbors::Neighbor* result) const;

  Status GetNeighborProtoNoMetadata(const pair<DatapointIndex, float> neighbor,
                                    const DatapointPtr<T>& query,
                                    NearestNeighbors::Neighbor* result) const;

  tensorflow::scann_ops::TypeTag TypeTag() const final {
    return TagForType<T>();
  }

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

  void set_compressed_dataset(
      shared_ptr<DenseDataset<uint8_t>> compressed_dataset) {
    compressed_dataset_ = std::move(compressed_dataset);
    docids_ = compressed_dataset_->docids();
  }

  bool compressed_reordering_enabled() const {
    return (reordering_helper_ &&
            reordering_helper_->name() == "CompressedReordering");
  }

  const ReorderingInterface<T>& reordering_helper() const {
    return *reordering_helper_;
  }

  class Mutator : public UntypedSingleMachineSearcherBase::UntypedMutator {
   public:
    virtual unique_ptr<MutationMetadata> ComputeMutationMetadata(
        const DatapointPtr<T>& dptr) const {
      return nullptr;
    }

    virtual vector<unique_ptr<MutationMetadata>> ComputeMutationMetadata(
        const TypedDataset<T>& batch) const {
      vector<unique_ptr<MutationMetadata>> result(batch.size());
      for (size_t i : IndicesOf(batch)) {
        result[i] = ComputeMutationMetadata(batch[i]);
      }
      return result;
    }

    virtual StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                                  string_view docid,
                                                  MutationMetadata* md) = 0;

    virtual StatusOr<DatapointIndex> UpdateDatapoint(
        const DatapointPtr<T>& dptr, string_view docid,
        MutationMetadata* md) = 0;
    virtual StatusOr<DatapointIndex> UpdateDatapoint(
        const DatapointPtr<T>& dptr, DatapointIndex index,
        MutationMetadata* md) = 0;

    bool LookupDatapointIndex(string_view docid,
                              DatapointIndex* index) const override;

    StatusOr<DatapointIndex> AddDatapointWithMetadata(
        const DatapointPtr<T>& dptr, const GenericFeatureVector& gfv,
        MutationMetadata* md = nullptr);
    StatusOr<DatapointIndex> UpdateDatapointWithMetadata(
        const DatapointPtr<T>& dptr, const GenericFeatureVector& gfv,
        DatapointIndex idx, MutationMetadata* md = nullptr);

    Status RemoveDatapointWithMetadata(DatapointIndex idx) final;

    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                          string_view docid) {
      return AddDatapoint(dptr, docid, nullptr);
    }
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             string_view docid) {
      return UpdateDatapoint(dptr, docid, nullptr);
    }
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             DatapointIndex index) {
      return UpdateDatapoint(dptr, index, nullptr);
    }

    Status PrepareForBaseMutation(SingleMachineSearcherBase<T>* searcher);

   protected:
    StatusOr<DatapointIndex> AddDatapointToBase(
        const DatapointPtr<T>& dptr, const DatapointPtr<uint8_t>& hashed,
        string_view docid);
    Status UpdateDatapointInBase(const DatapointPtr<T>& dptr,
                                 const DatapointPtr<uint8_t>& hashed,
                                 DatapointIndex idx);

    StatusOr<DatapointIndex> AddDatapointToBase(const DatapointPtr<T>& dptr,
                                                string_view docid) {
      return AddDatapointToBase(dptr, {}, docid);
    }
    Status UpdateDatapointInBase(const DatapointPtr<T>& dptr,
                                 DatapointIndex idx) {
      return UpdateDatapointInBase(dptr, {}, idx);
    }

    StatusOr<DatapointIndex> RemoveDatapointFromBase(DatapointIndex idx) final;
    void ReserveInBase(DatapointIndex num_datapoints) final;

   private:
    StatusOr<DatapointIndex> GetNextDatapointIndex() const;

    SingleMachineSearcherBase<T>* searcher_ = nullptr;

    typename TypedDataset<T>::Mutator* dataset_mutator_ = nullptr;

    typename TypedDataset<uint8_t>::Mutator* hashed_dataset_mutator_ = nullptr;

    DocidCollectionInterface::Mutator* docid_mutator_ = nullptr;

    typename ReorderingInterface<T>::Mutator* reordering_mutator_ = nullptr;
  };

  virtual StatusOr<typename SingleMachineSearcherBase::Mutator*> GetMutator()
      const {
    return FailedPreconditionError("Cannot be dynamically updated.");
  }
  StatusOr<typename UntypedSingleMachineSearcherBase::UntypedMutator*>
  GetUntypedMutator() const final {
    TF_ASSIGN_OR_RETURN(auto mutator, GetMutator());
    return mutator;
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

}  // namespace scann_ops
}  // namespace tensorflow

#endif
