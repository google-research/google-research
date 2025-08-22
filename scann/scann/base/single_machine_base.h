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



#ifndef SCANN_BASE_SINGLE_MACHINE_BASE_H_
#define SCANN_BASE_SINGLE_MACHINE_BASE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/metadata/metadata_getter.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/results.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/reordering_helper_interface.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

class UntypedSingleMachineSearcherBase;
using StatusOrSearcherUntyped =
    StatusOr<unique_ptr<UntypedSingleMachineSearcherBase>>;

template <typename T>
class SingleMachineSearcherBase;

template <typename T>
class BruteForceSearcher;

template <typename T>
void RetrainAndReindexFixup(UntypedSingleMachineSearcherBase* result,
                            const shared_ptr<Dataset>& dataset,
                            bool retraining_requires_dataset = false);

class UntypedSingleMachineSearcherBase {
 public:
  SCANN_DECLARE_IMMOBILE_CLASS(UntypedSingleMachineSearcherBase);

  virtual ~UntypedSingleMachineSearcherBase();

  virtual research_scann::TypeTag TypeTag() const = 0;

  virtual const Dataset* dataset() const = 0;

  const DenseDataset<uint8_t>* hashed_dataset() const {
    return hashed_dataset_.get();
  }

  StatusOr<string_view> GetDocid(DatapointIndex i) const;

  Status set_docids(shared_ptr<const DocidCollectionInterface> docids);

  shared_ptr<const DocidCollectionInterface> docids() const { return docids_; }

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
      std::vector<int64_t> datapoint_index_to_crowding_attribute,
      std::vector<std::string> crowding_dimension_names);
  Status EnableCrowding(
      shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute);
  Status EnableCrowding(
      shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute,
      shared_ptr<std::vector<std::string>> crowding_dimension_names);

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

  ConstSpan<std::string> crowding_dimension_names() const {
    ConstSpan<std::string> result;
    if (crowding_dimension_names_) {
      result = *crowding_dimension_names_;
    }
    return result;
  }

  void DisableCrowding() {
    DisableCrowdingImpl();
    datapoint_index_to_crowding_attribute_ = nullptr;
    crowding_dimension_names_ = nullptr;
  }

  StatusOr<DatapointIndex> DatasetSize() const;

  virtual int64_t num_active_dimensions() const {
    return (dataset() == nullptr) ? -1 : (dataset()->NumActiveDimensions());
  }

  virtual bool reordering_enabled() const = 0;

  virtual bool exact_reordering_enabled() const = 0;
  virtual bool fixed_point_reordering_enabled() const = 0;

  virtual DatapointIndex optimal_batch_size() const;

  virtual vector<uint32_t> SizeByPartition() const;

  virtual uint32_t NumPartitions() const { return 0; }

  class PrecomputedMutationArtifacts : public VirtualDestructor {};

  struct MutationOptions {
    PrecomputedMutationArtifacts* precomputed_mutation_artifacts = nullptr;

    bool reassignment_in_flight = false;
  };

  class UntypedMutator {
   public:
    virtual ~UntypedMutator() {}

    virtual Status RemoveDatapoint(string_view docid) = 0;

    virtual bool LookupDatapointIndex(string_view docid,
                                      DatapointIndex* index) const = 0;

    virtual void Reserve(size_t size) = 0;

    virtual Status RemoveDatapoint(DatapointIndex index) = 0;

    using DatapointIndexRenameFn =
        std::function<void(DatapointIndex old_idx, DatapointIndex new_idx)>;
    void AddOnDatapointIndexRenameFn(DatapointIndexRenameFn fn) {
      on_datapoint_index_rename_fns_.push_back(fn);
    }

   protected:
    struct MutateBaseOptions {
      std::optional<DatapointPtr<uint8_t>> hashed;
    };

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
  ExtractSingleMachineFactoryOptions() = 0;

  Status GetNeighborProtoNoMetadata(pair<DatapointIndex, float> neighbor,
                                    NearestNeighbors::Neighbor* result) const;

  std::optional<ScannConfig> config() const { return config_; }
  void set_config(ScannConfig config) { config_ = config; }

 protected:
  virtual bool impl_needs_dataset() const;

  virtual bool impl_needs_hashed_dataset() const;

  virtual Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
      ConstSpan<std::string> crowding_dimension_names) {
    return OkStatus();
  }

  virtual void DisableCrowdingImpl() {}

 private:
  UntypedSingleMachineSearcherBase(
      shared_ptr<const DenseDataset<uint8_t>> hashed_dataset,
      int32_t default_pre_reordering_num_neighbors,
      float default_pre_reordering_epsilon);

  UntypedSingleMachineSearcherBase() = default;

  template <typename ResultElem>
  Status ValidateFindNeighborsBatched(const Dataset& queries,
                                      ConstSpan<SearchParameters> params,
                                      MutableSpan<ResultElem> results) const;

  Status SortAndDropResults(NNResultsVector* result,
                            const SearchParameters& params) const;

  shared_ptr<const DocidCollectionInterface> docids_;

  shared_ptr<std::vector<std::string>> crowding_dimension_names_ = {};

  SearchParameters default_search_parameters_;

  std::optional<ScannConfig> config_ = std::nullopt;

  bool mutator_outstanding_ = false;

  bool retraining_requires_dataset_ = false;

  bool exact_reordering_enabled_ = false;

  int64_t creation_timestamp_ = numeric_limits<int64_t>::min();

  shared_ptr<std::vector<int64_t>> datapoint_index_to_crowding_attribute_ = {};

  shared_ptr<UntypedMetadataGetter> metadata_getter_;

  template <typename T>
  friend class SingleMachineSearcherBase;

  template <typename T>
  friend void RetrainAndReindexFixup(UntypedSingleMachineSearcherBase* result,
                                     const shared_ptr<Dataset>& dataset,
                                     bool retraining_requires_dataset);

  template <typename T>
  friend StatusOrSearcherUntyped RetrainAndReindexSearcherImpl(
      UntypedSingleMachineSearcherBase* untyped_searcher,
      absl::Mutex* searcher_pointer_mutex, ScannConfig config,
      shared_ptr<ThreadPool> parallelization_pool);

  shared_ptr<const DenseDataset<uint8_t>> hashed_dataset_ = nullptr;
};

template <typename T>
class SingleMachineSearcherBase : public UntypedSingleMachineSearcherBase {
 public:
  using DataType = T;

  SCANN_DECLARE_IMMOBILE_CLASS(SingleMachineSearcherBase);

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
  Status FindNeighborsBatchedNoSortNoExactReorder(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<FastTopNeighbors<float>*> results,
      ConstSpan<DatapointIndex> datapoint_index_lookup) const;

  virtual Status PreprocessQueryIntoParamsUnlocked(
      const DatapointPtr<T>& query, SearchParameters& search_params) const {
    search_params.set_unlocked_query_preprocessing_results(nullptr);
    return OkStatus();
  }

  virtual StatusOr<shared_ptr<const DenseDataset<float>>>
  SharedFloatDatasetIfNeeded();

  virtual StatusOr<shared_ptr<const DenseDataset<float>>>
  ReconstructFloatDataset();

  Status GetNeighborProto(pair<DatapointIndex, float> neighbor,
                          const DatapointPtr<T>& query,
                          NearestNeighbors::Neighbor* result) const;

  using UntypedSingleMachineSearcherBase::GetNeighborProtoNoMetadata;
  Status GetNeighborProtoNoMetadata(pair<DatapointIndex, float> neighbor,
                                    const DatapointPtr<T>& query,
                                    NearestNeighbors::Neighbor* result) const {
    return GetNeighborProtoNoMetadata(neighbor, result);
  }

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
    exact_reordering_enabled_ =
        (reordering_helper_ && reordering_helper_->name() == "ExactReordering");
    default_search_parameters_.set_post_reordering_num_neighbors(
        default_post_reordering_num_neighbors);
    default_search_parameters_.set_post_reordering_epsilon(
        default_post_reordering_epsilon);
  }

  void DisableReordering() {
    reordering_helper_ = nullptr;
    exact_reordering_enabled_ = false;
  }

  bool reordering_enabled() const final {
    return reordering_helper_ != nullptr;
  }

  void DisableExactReordering() { DisableReordering(); }

  virtual StatusOr<const SingleMachineSearcherBase<T>*>
  CreateBruteForceSearcher(
      const DistanceMeasureConfig& distance_config,
      unique_ptr<SingleMachineSearcherBase<T>>* storage) const;

  bool exact_reordering_enabled() const final {
    return exact_reordering_enabled_;
  }

  bool fixed_point_reordering_enabled() const final;

  const ReorderingInterface<T>& reordering_helper() const {
    return *reordering_helper_;
  }

  class Mutator : public UntypedSingleMachineSearcherBase::UntypedMutator {
   public:
    virtual unique_ptr<PrecomputedMutationArtifacts>
    ComputePrecomputedMutationArtifacts(const DatapointPtr<T>& dptr) const {
      return nullptr;
    }

    virtual vector<unique_ptr<PrecomputedMutationArtifacts>>
    ComputePrecomputedMutationArtifacts(const TypedDataset<T>& batch) const {
      vector<unique_ptr<PrecomputedMutationArtifacts>> result(batch.size());
      for (size_t i : IndicesOf(batch)) {
        result[i] = ComputePrecomputedMutationArtifacts(batch[i]);
      }
      return result;
    }

    virtual vector<unique_ptr<PrecomputedMutationArtifacts>>
    ComputePrecomputedMutationArtifacts(
        const TypedDataset<T>& batch,
        shared_ptr<ThreadPool> parallelization_pool) const {
      return ComputePrecomputedMutationArtifacts(batch);
    }
    virtual shared_ptr<ThreadPool> mutation_threadpool() const {
      return mutation_threadpool_;
    }
    virtual void set_mutation_threadpool(shared_ptr<ThreadPool> pool) {
      mutation_threadpool_ = pool;
    }

    virtual StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex i) const {
      return UnimplementedError("GetDatapoint not implemented.");
    }

    virtual StatusOr<DatapointIndex> AddDatapoint(
        const DatapointPtr<T>& dptr, string_view docid,
        const MutationOptions& mo) = 0;

    virtual StatusOr<DatapointIndex> UpdateDatapoint(
        const DatapointPtr<T>& dptr, string_view docid,
        const MutationOptions& mo) = 0;
    virtual StatusOr<DatapointIndex> UpdateDatapoint(
        const DatapointPtr<T>& dptr, DatapointIndex index,
        const MutationOptions& mo) = 0;

    virtual Status EnableIncrementalTraining(const ScannConfig& config) {
      return UnimplementedError("EnableIncrementalTraining not implemented.");
    }
    bool LookupDatapointIndex(string_view docid,
                              DatapointIndex* index) const override;

    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                          string_view docid) {
      return AddDatapoint(dptr, docid, MutationOptions());
    }
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             string_view docid) {
      return UpdateDatapoint(dptr, docid, MutationOptions());
    }
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             DatapointIndex index) {
      return UpdateDatapoint(dptr, index, MutationOptions());
    }

    Status PrepareForBaseMutation(SingleMachineSearcherBase<T>* searcher);

    virtual StatusOr<std::optional<ScannConfig>> IncrementalMaintenance() {
      return std::nullopt;
    }

    Status ValidateForAdd(const DatapointPtr<T>& dptr, string_view docid,
                          const MutationOptions& mo) const;
    Status ValidateForUpdate(const DatapointPtr<T>& dptr, DatapointIndex idx,
                             const MutationOptions& mo) const;
    Status ValidateForRemove(DatapointIndex idx) const;

   protected:
    StatusOr<Datapoint<T>> GetDatapointFromBase(DatapointIndex i) const;
    StatusOr<DatapointIndex> AddDatapointToBase(const DatapointPtr<T>& dptr,
                                                string_view docid,
                                                const MutateBaseOptions& opts);
    Status UpdateDatapointInBase(const DatapointPtr<T>& dptr,
                                 DatapointIndex idx,
                                 const MutateBaseOptions& opts);
    StatusOr<DatapointIndex> RemoveDatapointFromBase(DatapointIndex idx) final;
    void ReserveInBase(DatapointIndex num_datapoints) final;

   private:
    StatusOr<DatapointIndex> GetNextDatapointIndex() const;

    Status CheckAddDatapointToBaseOptions(const MutateBaseOptions& opts) const;

    Status ValidateForUpdateOrAdd(const DatapointPtr<T>& dptr,
                                  string_view docid,
                                  const MutationOptions& mo) const;

    SingleMachineSearcherBase<T>* searcher_ = nullptr;

    typename TypedDataset<T>::Mutator* dataset_mutator_ = nullptr;

    typename TypedDataset<uint8_t>::Mutator* hashed_dataset_mutator_ = nullptr;

    DocidCollectionInterface::Mutator* docid_mutator_ = nullptr;

    typename ReorderingInterface<T>::Mutator* reordering_mutator_ = nullptr;

    shared_ptr<ThreadPool> mutation_threadpool_ = nullptr;
  };

  virtual StatusOr<typename SingleMachineSearcherBase::Mutator*> GetMutator()
      const {
    return FailedPreconditionError("Cannot be dynamically updated.");
  }
  StatusOr<typename UntypedSingleMachineSearcherBase::UntypedMutator*>
  GetUntypedMutator() const final {
    SCANN_ASSIGN_OR_RETURN(auto mutator, GetMutator());
    return mutator;
  }

  struct HealthStats {
    double partition_weighted_avg_relative_imbalance = 0;

    double partition_avg_relative_positive_imbalance = 0;

    double avg_quantization_error = 0;

    uint64_t sum_partition_sizes = 0;

    bool operator==(const HealthStats& rhs) const {
      return partition_weighted_avg_relative_imbalance ==
                 rhs.partition_weighted_avg_relative_imbalance &&
             partition_avg_relative_positive_imbalance ==
                 rhs.partition_avg_relative_positive_imbalance &&
             sum_partition_sizes == rhs.sum_partition_sizes &&

             abs(avg_quantization_error - rhs.avg_quantization_error) < 1e-5;
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const HealthStats& s) {
      absl::Format(&sink,
                   "{partition_weighted_avg_relative_imbalance = %f; "
                   "partition_avg_relative_positive_imbalance = %f; "
                   "avg_quantization_error = %f; sum_partition_sizes = %u}",
                   s.partition_weighted_avg_relative_imbalance,
                   s.partition_avg_relative_positive_imbalance,
                   s.avg_quantization_error, s.sum_partition_sizes);
    }
  };

  virtual StatusOr<HealthStats> GetHealthStats() const { return HealthStats(); }

  virtual Status InitializeHealthStats() { return OkStatus(); }

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

  virtual Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<FastTopNeighbors<float>*> results,
      ConstSpan<DatapointIndex> datapoint_index_mapping) const;

  Status SampleRandomNeighbors(const DatapointPtr<T>& query,
                               const SearchParameters& params,
                               NNResultsVector* result) const;

  virtual Status PropagateDistances(const DatapointPtr<T>& query,
                                    const SearchParameters& params,
                                    NNResultsVector* result) const;

 private:
  Status PopulateDefaultParameters(const ScannConfig& config);
  Status BaseInitImpl();

  Status ReorderResults(const DatapointPtr<T>& query,
                        const SearchParameters& params,
                        NNResultsVector* result) const;

  template <typename TopN>
  Status SampleRandomNeighborsImpl(const SearchParameters& params,
                                   TopN* top_n_ptr) const;

  shared_ptr<const TypedDataset<T>> dataset_ = nullptr;

  shared_ptr<const ReorderingInterface<T>> reordering_helper_ = nullptr;

  friend class Mutator;

  friend void RetrainAndReindexFixup<T>(
      UntypedSingleMachineSearcherBase* result,
      const shared_ptr<Dataset>& dataset, bool retraining_requires_dataset);

  friend StatusOrSearcherUntyped RetrainAndReindexSearcherImpl<T>(
      UntypedSingleMachineSearcherBase* untyped_searcher,
      absl::Mutex* searcher_pointer_mutex, ScannConfig config,
      shared_ptr<ThreadPool> parallelization_pool);

  friend class BruteForceSearcher<T>;
};

template <typename T>
void RetrainAndReindexFixup(UntypedSingleMachineSearcherBase* result,
                            const shared_ptr<Dataset>& dataset,
                            bool retraining_requires_dataset) {
  result->retraining_requires_dataset_ = retraining_requires_dataset;
  down_cast<SingleMachineSearcherBase<T>*>(result)->dataset_ =
      std::dynamic_pointer_cast<TypedDataset<T>>(dataset);

  result->docids_ = dataset->docids();
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, SingleMachineSearcherBase);

}  // namespace research_scann

#endif
