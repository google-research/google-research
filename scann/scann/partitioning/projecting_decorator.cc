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



#include "scann/partitioning/projecting_decorator.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/projection/pca_projection.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"

namespace research_scann {

template <typename T>
ProjectingDecoratorInterface<T>::~ProjectingDecoratorInterface() = default;

template <typename Base, typename T>
ProjectingDecoratorBase<Base, T>::ProjectingDecoratorBase(
    shared_ptr<const Projection<T>> projection,
    unique_ptr<Partitioner<float>> partitioner)
    : projection_(std::move(projection)), partitioner_(std::move(partitioner)) {
  DCHECK(projection_);
  DCHECK(partitioner_);

  CHECK(!dynamic_cast<ProjectingDecoratorInterface<float>*>(partitioner_.get()))
      << typeid(*partitioner_).name();
  this->set_tokenization_mode_no_hook(partitioner_->tokenization_mode());
}

template <typename Base, typename T>
void ProjectingDecoratorBase<Base, T>::CopyToProto(
    SerializedPartitioner* result) const {
  partitioner_->CopyToProto(result);
  result->set_uses_projection(true);

  auto pca = dynamic_cast<const PcaProjection<T>*>(projection_.get());
  if (!pca) return;

  result->clear_serialized_projection();
  shared_ptr<const TypedDataset<float>> eigenvectors =
      projection_->GetDirections().value();
  for (DatapointPtr<float> eigenvector : *eigenvectors) {
    *result->mutable_serialized_projection()->add_rotation_vec() =
        eigenvector.ToGfv();
  }
}

template <typename Base, typename T>
Normalization ProjectingDecoratorBase<Base, T>::NormalizationRequired() const {
  return NONE;
}

template <typename Base, typename T>
Status ProjectingDecoratorBase<Base, T>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, int32_t* result) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<float> projected, ProjectAndNormalize(dptr));
  return partitioner_->TokenForDatapoint(projected.ToPtr(), result);
}

template <typename Base, typename T>
Status ProjectingDecoratorBase<Base, T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& dptr, vector<int32_t>* result) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<float> projected, ProjectAndNormalize(dptr));
  return partitioner_->TokensForDatapointWithSpilling(projected.ToPtr(),
                                                      result);
}

template <typename Base, typename T>
int32_t ProjectingDecoratorBase<Base, T>::n_tokens() const {
  return partitioner_->n_tokens();
}

template <typename T>
Status KMeansTreeProjectingDecorator<T>::TokensForDatapointWithSpilling(
    const DatapointPtr<T>& dptr, int32_t max_centers_override,
    vector<pair<DatapointIndex, float>>* result) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<float> projected,
                         this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->TokensForDatapointWithSpilling(
      projected.ToPtr(), max_centers_override, result);
}

template <typename T>
StatusOrPtr<TypedDataset<float>>
KMeansTreeProjectingDecorator<T>::CreateProjectedDataset(
    const TypedDataset<T>& queries) const {
  if (queries.empty()) return {nullptr};
  unique_ptr<TypedDataset<float>> projected_ds;
  for (size_t i : IndicesOf(queries)) {
    SCANN_ASSIGN_OR_RETURN(auto projected,
                           this->ProjectAndNormalize(queries[i]));
    if (!projected_ds) {
      if (projected.IsSparse()) {
        projected_ds = make_unique<SparseDataset<float>>();
      } else {
        projected_ds = make_unique<DenseDataset<float>>();
      }
      projected_ds->set_dimensionality(projected.dimensionality());
      projected_ds->Reserve(queries.size());
    }
    SCANN_RETURN_IF_ERROR(projected_ds->Append(projected.ToPtr(), ""));
  }
  return projected_ds;
}

template <typename T>
Status KMeansTreeProjectingDecorator<T>::TokenForDatapointBatched(
    const TypedDataset<T>& queries,
    std::vector<pair<DatapointIndex, float>>* results, ThreadPool* pool) const {
  if (queries.empty()) {
    results->clear();
    return OkStatus();
  }
  SCANN_ASSIGN_OR_RETURN(unique_ptr<TypedDataset<float>> projected_ds,
                         CreateProjectedDataset(queries));
  return base_kmeans_tree_partitioner()->TokenForDatapointBatched(
      *projected_ds, results, pool);
}

template <typename T>
Status KMeansTreeProjectingDecorator<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, MutableSpan<std::vector<int32_t>> results,
    ThreadPool* pool) const {
  if (queries.empty()) return OkStatus();
  SCANN_ASSIGN_OR_RETURN(unique_ptr<TypedDataset<float>> projected_ds,
                         CreateProjectedDataset(queries));
  DLOG(INFO) << "projected_ds: " << projected_ds->dimensionality();
  return base_kmeans_tree_partitioner()->TokensForDatapointWithSpillingBatched(
      *projected_ds, results, pool);
}

template <typename T>
Status KMeansTreeProjectingDecorator<T>::TokensForDatapointWithSpillingBatched(
    const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
    MutableSpan<std::vector<pair<DatapointIndex, float>>> results,
    ThreadPool* pool) const {
  if (queries.empty()) return OkStatus();
  SCANN_ASSIGN_OR_RETURN(unique_ptr<TypedDataset<float>> projected_ds,
                         CreateProjectedDataset(queries));
  return base_kmeans_tree_partitioner()->TokensForDatapointWithSpillingBatched(
      *projected_ds, max_centers_override, results, pool);
}

template <typename T>
Status KMeansTreeProjectingDecorator<T>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, pair<DatapointIndex, float>* result) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<float> projected,
                         this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->TokenForDatapoint(projected.ToPtr(),
                                                           result);
}

template <typename T>
StatusOr<Datapoint<float>> KMeansTreeProjectingDecorator<T>::ResidualizeToFloat(
    const DatapointPtr<T>& dptr, int32_t token) const {
  SCANN_ASSIGN_OR_RETURN(Datapoint<float> projected,
                         this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->ResidualizeToFloat(projected.ToPtr(),
                                                            token);
}

SCANN_INSTANTIATE_PROJECTING_DECORATOR_ALL();

}  // namespace research_scann
