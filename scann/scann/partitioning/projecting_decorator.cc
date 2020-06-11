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



#include "scann/partitioning/projecting_decorator.h"

#include "scann/utils/datapoint_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace scann_ops {

template <typename Base, typename T, typename ProjectionType>
ProjectingDecoratorBase<Base, T, ProjectionType>::ProjectingDecoratorBase(
    shared_ptr<const Projection<T>> projection,
    unique_ptr<Partitioner<ProjectionType>> partitioner)
    : projection_(std::move(projection)), partitioner_(std::move(partitioner)) {
  DCHECK(projection_);
  DCHECK(partitioner_);
  this->set_tokenization_mode_no_hook(partitioner_->tokenization_mode());
}

template <typename Base, typename T, typename ProjectionType>
void ProjectingDecoratorBase<Base, T, ProjectionType>::CopyToProto(
    SerializedPartitioner* result) const {
  partitioner_->CopyToProto(result);
  result->set_uses_projection(true);
}

template <typename Base, typename T, typename ProjectionType>
Normalization ProjectingDecoratorBase<
    Base, T, ProjectionType>::NormalizationRequired() const {
  return NONE;
}

template <typename Base, typename T, typename ProjectionType>
Status ProjectingDecoratorBase<Base, T, ProjectionType>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, int32_t* result) const {
  TF_ASSIGN_OR_RETURN(Datapoint<ProjectionType> projected,
                      ProjectAndNormalize(dptr));
  return partitioner_->TokenForDatapoint(projected.ToPtr(), result);
}

template <typename Base, typename T, typename ProjectionType>
Status ProjectingDecoratorBase<Base, T, ProjectionType>::
    TokensForDatapointWithSpilling(const DatapointPtr<T>& dptr,
                                   vector<int32_t>* result) const {
  TF_ASSIGN_OR_RETURN(Datapoint<ProjectionType> projected,
                      ProjectAndNormalize(dptr));
  return partitioner_->TokensForDatapointWithSpilling(projected.ToPtr(),
                                                      result);
}

template <typename Base, typename T, typename ProjectionType>
int32_t ProjectingDecoratorBase<Base, T, ProjectionType>::n_tokens() const {
  return partitioner_->n_tokens();
}

template <typename T, typename ProjectionType>
Status KMeansTreeProjectingDecorator<T, ProjectionType>::
    TokensForDatapointWithSpilling(
        const DatapointPtr<T>& dptr, int32_t max_centers_override,
        vector<KMeansTreeSearchResult>* result) const {
  TF_ASSIGN_OR_RETURN(Datapoint<ProjectionType> projected,
                      this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->TokensForDatapointWithSpilling(
      projected.ToPtr(), max_centers_override, result);
}

template <typename T, typename ProjectionType>
Status KMeansTreeProjectingDecorator<T, ProjectionType>::
    TokensForDatapointWithSpillingBatched(
        const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
        MutableSpan<std::vector<KMeansTreeSearchResult>> results) const {
  if (queries.empty()) return OkStatus();
  unique_ptr<TypedDataset<ProjectionType>> projected_ds;
  for (size_t i : IndicesOf(queries)) {
    TF_ASSIGN_OR_RETURN(auto projected, this->ProjectAndNormalize(queries[i]));
    if (!projected_ds) {
      if (projected.IsSparse()) {
        projected_ds = make_unique<SparseDataset<ProjectionType>>();
      } else {
        projected_ds = make_unique<DenseDataset<ProjectionType>>();
      }
      projected_ds->set_dimensionality(projected.dimensionality());
      projected_ds->Reserve(queries.size());
    }
    SCANN_RETURN_IF_ERROR(projected_ds->Append(projected.ToPtr(), ""));
  }
  return base_kmeans_tree_partitioner()->TokensForDatapointWithSpillingBatched(
      *projected_ds, max_centers_override, results);
}

template <typename T, typename ProjectionType>
Status KMeansTreeProjectingDecorator<T, ProjectionType>::TokenForDatapoint(
    const DatapointPtr<T>& dptr, KMeansTreeSearchResult* result) const {
  TF_ASSIGN_OR_RETURN(Datapoint<ProjectionType> projected,
                      this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->TokenForDatapoint(projected.ToPtr(),
                                                           result);
}

template <typename T, typename ProjectionType>
StatusOr<Datapoint<float>>
KMeansTreeProjectingDecorator<T, ProjectionType>::ResidualizeToFloat(
    const DatapointPtr<T>& dptr, int32_t token,
    bool normalize_residual_by_cluster_stdev) const {
  TF_ASSIGN_OR_RETURN(Datapoint<ProjectionType> projected,
                      this->ProjectAndNormalize(dptr));
  return base_kmeans_tree_partitioner()->ResidualizeToFloat(
      projected.ToPtr(), token, normalize_residual_by_cluster_stdev);
}

#define INSTANTIATE_PROJECTING_DECORATOR(T)                               \
  template class ProjectingDecoratorBase<Partitioner<T>, T, float>;       \
  template class ProjectingDecoratorBase<Partitioner<T>, T, double>;      \
  template class ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T, \
                                         float>;                          \
  template class ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T, \
                                         double>;                         \
  template class GenericProjectingDecorator<T, float>;                    \
  template class GenericProjectingDecorator<T, double>;                   \
  template class KMeansTreeProjectingDecorator<T, float>;                 \
  template class KMeansTreeProjectingDecorator<T, double>;

INSTANTIATE_PROJECTING_DECORATOR(int8_t);
INSTANTIATE_PROJECTING_DECORATOR(uint8_t);
INSTANTIATE_PROJECTING_DECORATOR(int16_t);
INSTANTIATE_PROJECTING_DECORATOR(uint16_t);
INSTANTIATE_PROJECTING_DECORATOR(int32_t);
INSTANTIATE_PROJECTING_DECORATOR(uint32_t);
INSTANTIATE_PROJECTING_DECORATOR(int64_t);
INSTANTIATE_PROJECTING_DECORATOR(unsigned long long_t);
INSTANTIATE_PROJECTING_DECORATOR(float);
INSTANTIATE_PROJECTING_DECORATOR(double);

}  // namespace scann_ops
}  // namespace tensorflow
