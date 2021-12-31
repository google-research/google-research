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



#ifndef SCANN_PARTITIONING_PROJECTING_DECORATOR_H_
#define SCANN_PARTITIONING_PROJECTING_DECORATOR_H_

#include <cstdint>
#include <type_traits>

#include "absl/memory/memory.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/partitioning/projecting_decorator.h"
#include "scann/projection/projection_base.h"

namespace research_scann {

template <typename Base, typename T, typename ProjectionType>
class ProjectingDecoratorBase : public Base {
 public:
  ProjectingDecoratorBase(shared_ptr<const Projection<T>> projection,
                          unique_ptr<Partitioner<ProjectionType>> partitioner);

  void CopyToProto(SerializedPartitioner* result) const override;

  Normalization NormalizationRequired() const override;

  using Base::TokenForDatapoint;
  using Base::TokensForDatapointWithSpilling;
  using Base::TokensForDatapointWithSpillingBatched;

  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           int32_t* result) const override;

  Status TokensForDatapointWithSpilling(const DatapointPtr<T>& dptr,
                                        vector<int32_t>* result) const override;

  int32_t n_tokens() const override;

  Partitioner<ProjectionType>* base_partitioner() const {
    return partitioner_.get();
  }

  const shared_ptr<const Projection<T>>& projection() const {
    return projection_;
  }

  StatusOr<Datapoint<ProjectionType>> ProjectAndNormalize(
      const DatapointPtr<T>& dptr) const {
    Datapoint<ProjectionType> projected;
    SCANN_RETURN_IF_ERROR(projection_->ProjectInput(dptr, &projected));
    SCANN_RETURN_IF_ERROR(NormalizeByTag(
        base_partitioner()->NormalizationRequired(), &projected));
    return projected;
  }

 private:
  static_assert(IsFloatingType<ProjectionType>(),
                "ProjectionType must be float/double.");
  void OnSetTokenizationMode() final {
    partitioner_->set_tokenization_mode(this->tokenization_mode());
  }

  shared_ptr<const Projection<T>> projection_;

  unique_ptr<Partitioner<ProjectionType>> partitioner_;
};

template <typename T, typename ProjectionType = double>
class GenericProjectingDecorator final
    : public ProjectingDecoratorBase<Partitioner<T>, T, ProjectionType> {
 public:
  using Base = ProjectingDecoratorBase<Partitioner<T>, T, ProjectionType>;

  GenericProjectingDecorator(
      shared_ptr<const Projection<T>> projection,
      unique_ptr<Partitioner<ProjectionType>> partitioner)
      : Base(std::move(projection), std::move(partitioner)) {}

  unique_ptr<Partitioner<T>> Clone() const final {
    return unique_ptr<Partitioner<T>>(
        new GenericProjectingDecorator<T, ProjectionType>(
            this->projection(), this->base_partitioner()->Clone()));
  }
};

template <typename T, typename ProjectionType = double>
class KMeansTreeProjectingDecorator final
    : public ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T,
                                     ProjectionType> {
 public:
  using Base =
      ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T, ProjectionType>;
  KMeansTreeProjectingDecorator(
      shared_ptr<const Projection<T>> projection,
      unique_ptr<KMeansTreeLikePartitioner<ProjectionType>> partitioner)
      : Base(std::move(projection),
             unique_ptr<Partitioner<ProjectionType>>(partitioner.release())) {}

  KMeansTreeLikePartitioner<ProjectionType>* base_kmeans_tree_partitioner()
      const {
    return down_cast<KMeansTreeLikePartitioner<ProjectionType>*>(
        this->base_partitioner());
  }

  unique_ptr<Partitioner<T>> Clone() const final {
    auto partitioner_clone = base_kmeans_tree_partitioner()->Clone();
    auto downcast_partitioner =
        down_cast<KMeansTreeLikePartitioner<ProjectionType>*>(
            partitioner_clone.release());
    return make_unique<KMeansTreeProjectingDecorator<T, ProjectionType>>(
        this->projection(), absl::WrapUnique(downcast_partitioner));
  }

  const shared_ptr<const DistanceMeasure>& database_tokenization_distance()
      const final {
    return base_kmeans_tree_partitioner()->database_tokenization_distance();
  }

  const shared_ptr<const DistanceMeasure>& query_tokenization_distance()
      const final {
    return base_kmeans_tree_partitioner()->query_tokenization_distance();
  }

  const shared_ptr<const KMeansTree>& kmeans_tree() const final {
    return base_kmeans_tree_partitioner()->kmeans_tree();
  }

  using Partitioner<T>::TokensForDatapointWithSpilling;
  using Partitioner<T>::TokensForDatapointWithSpillingBatched;
  using Partitioner<T>::TokenForDatapoint;

  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      vector<KMeansTreeSearchResult>* result) const final;
  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<KMeansTreeSearchResult>> results) const final;
  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           KMeansTreeSearchResult* result) const final;
  StatusOr<Datapoint<float>> ResidualizeToFloat(
      const DatapointPtr<T>& dptr, int32_t token,
      bool normalize_residual_by_cluster_stdev) const final;
};

}  // namespace research_scann

#endif
