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



#ifndef SCANN_PARTITIONING_PROJECTING_DECORATOR_H_
#define SCANN_PARTITIONING_PROJECTING_DECORATOR_H_

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/partitioning/partitioner_base.h"
#include "scann/projection/projection_base.h"

namespace research_scann {

template <typename T>
class ProjectingDecoratorInterface {
 public:
  virtual ~ProjectingDecoratorInterface();
  virtual Partitioner<float>* base_partitioner() const = 0;
  virtual const shared_ptr<const Projection<T>>& projection() const = 0;
  virtual StatusOr<Datapoint<float>> ProjectAndNormalize(
      const DatapointPtr<T>& dptr) const = 0;
};

template <typename Base, typename T>
class ProjectingDecoratorBase : public Base,
                                public ProjectingDecoratorInterface<T> {
 public:
  ProjectingDecoratorBase(shared_ptr<const Projection<T>> projection,
                          unique_ptr<Partitioner<float>> partitioner);

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

  Partitioner<float>* base_partitioner() const override {
    return partitioner_.get();
  }

  const shared_ptr<const Projection<T>>& projection() const final {
    return projection_;
  }

  StatusOr<Datapoint<float>> ProjectAndNormalize(
      const DatapointPtr<T>& dptr) const override {
    Datapoint<float> projected;
    SCANN_RETURN_IF_ERROR(projection_->ProjectInput(dptr, &projected));
    SCANN_RETURN_IF_ERROR(NormalizeByTag(
        base_partitioner()->NormalizationRequired(), &projected));
    return projected;
  }

 private:
  void OnSetTokenizationMode() final {
    partitioner_->set_tokenization_mode(this->tokenization_mode());
  }

  shared_ptr<const Projection<T>> projection_;

  unique_ptr<Partitioner<float>> partitioner_;
};

template <typename T>
class GenericProjectingDecorator final
    : public ProjectingDecoratorBase<Partitioner<T>, T> {
 public:
  using Base = ProjectingDecoratorBase<Partitioner<T>, T>;

  GenericProjectingDecorator(shared_ptr<const Projection<T>> projection,
                             unique_ptr<Partitioner<float>> partitioner)
      : Base(std::move(projection), std::move(partitioner)) {}

  unique_ptr<Partitioner<T>> Clone() const final {
    return unique_ptr<Partitioner<T>>(new GenericProjectingDecorator<T>(
        this->projection(), this->base_partitioner()->Clone()));
  }
};

template <typename T>
class KMeansTreeProjectingDecorator final
    : public ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T> {
 public:
  using Base = ProjectingDecoratorBase<KMeansTreeLikePartitioner<T>, T>;
  KMeansTreeProjectingDecorator(
      shared_ptr<const Projection<T>> projection,
      unique_ptr<KMeansTreeLikePartitioner<float>> partitioner)
      : Base(std::move(projection),
             unique_ptr<Partitioner<float>>(partitioner.release())) {}

  KMeansTreeLikePartitioner<float>* base_kmeans_tree_partitioner() const {
    return down_cast<KMeansTreeLikePartitioner<float>*>(
        this->base_partitioner());
  }

  unique_ptr<Partitioner<T>> Clone() const final {
    auto partitioner_clone = base_kmeans_tree_partitioner()->Clone();
    auto downcast_partitioner = down_cast<KMeansTreeLikePartitioner<float>*>(
        partitioner_clone.release());
    return make_unique<KMeansTreeProjectingDecorator<T>>(
        this->projection(), absl::WrapUnique(downcast_partitioner));
  }

  const shared_ptr<const DistanceMeasure>& query_tokenization_distance()
      const final {
    return base_kmeans_tree_partitioner()->query_tokenization_distance();
  }

  const shared_ptr<const KMeansTree>& kmeans_tree() const final {
    return base_kmeans_tree_partitioner()->kmeans_tree();
  }

  const DenseDataset<float>& LeafCenters() const final {
    return base_kmeans_tree_partitioner()->LeafCenters();
  }

  using Partitioner<T>::TokensForDatapointWithSpilling;
  using Partitioner<T>::TokenForDatapoint;

  Status TokensForDatapointWithSpilling(
      const DatapointPtr<T>& dptr, int32_t max_centers_override,
      vector<pair<DatapointIndex, float>>* result) const final;
  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, MutableSpan<std::vector<int32_t>> results,
      ThreadPool* pool = nullptr) const final;
  Status TokensForDatapointWithSpillingBatched(
      const TypedDataset<T>& queries, ConstSpan<int32_t> max_centers_override,
      MutableSpan<std::vector<pair<DatapointIndex, float>>> results,
      ThreadPool* pool = nullptr) const final;
  Status TokenForDatapoint(const DatapointPtr<T>& dptr,
                           pair<DatapointIndex, float>* result) const final;
  Status TokenForDatapointBatched(
      const TypedDataset<T>& queries,
      std::vector<pair<DatapointIndex, float>>* result,
      ThreadPool* pool) const final;
  StatusOr<Datapoint<float>> ResidualizeToFloat(const DatapointPtr<T>& dptr,
                                                int32_t token) const final;

  uint32_t query_spilling_max_centers() const final {
    return base_kmeans_tree_partitioner()->query_spilling_max_centers();
  }

 private:
  StatusOrPtr<TypedDataset<float>> CreateProjectedDataset(
      const TypedDataset<T>& queries) const;
};

#define SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, T)           \
  extern_keyword template class ProjectingDecoratorInterface<T>;            \
  extern_keyword template class ProjectingDecoratorBase<Partitioner<T>, T>; \
  extern_keyword template class ProjectingDecoratorBase<                    \
      KMeansTreeLikePartitioner<T>, T>;                                     \
  extern_keyword template class GenericProjectingDecorator<T>;              \
  extern_keyword template class KMeansTreeProjectingDecorator<T>;

#define SCANN_INSTANTIATE_PROJECTING_DECORATOR_ALL(extern_keyword)  \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, int8_t);   \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, uint8_t);  \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, int16_t);  \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, int32_t);  \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, uint32_t); \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, int64_t);  \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, float);    \
  SCANN_INSTANTIATE_PROJECTING_DECORATOR(extern_keyword, double);

SCANN_INSTANTIATE_PROJECTING_DECORATOR_ALL(extern);

}  // namespace research_scann

#endif
