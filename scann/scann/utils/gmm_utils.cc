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

#include "scann/utils/gmm_utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "absl/base/internal/endian.h"
#include "absl/container/flat_hash_set.h"
#include "absl/numeric/bits.h"
#include "absl/random/discrete_distribution.h"
#include "absl/random/distributions.h"
#include "absl/time/time.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {

GmmUtils::GmmUtils(shared_ptr<const DistanceMeasure> distance, Options opts)
    : distance_(std::move(distance)), opts_(opts), random_(opts_.seed) {}

namespace {

void BiasDistances(double bias, MutableSpan<double> distances) {
  if (bias == 0.0) return;
  for (size_t j : Seq(distances.size())) {
    distances[j] += bias;
  }
}

template <typename FloatT>
Status VerifyDatasetAllFinite(const DenseDatasetView<FloatT>& dataset) {
  size_t dim = dataset.dimensionality();
  for (size_t i : IndicesOf(dataset)) {
    SCANN_RETURN_IF_ERROR(
        VerifyAllFinite(MakeConstSpan(dataset.GetPtr(i), dim)))
        << "Invalid data at index: " << i;
  }
  return OkStatus();
}

DatapointIndex GetSample(MTRandom* random, ConstSpan<double> distances,
                         double distances_sum, bool is_first) {
  if (distances_sum <= 0.0 || std::isnan(distances_sum)) {
    VLOG(1) << StrFormat(
        "All %d points are zero distance from the centers (distances_sum = "
        "%f).",
        distances.size(), distances_sum);
    if (is_first) {
      LOG_EVERY_N(WARNING, 1000000) << StrFormat(
          "All %d points are exactly the same. (distances_sum = %f)",
          distances.size(), distances_sum);
    }
    return distances.size() - 1;
  }

  const double target = absl::Uniform<double>(*random, 0.0, distances_sum);

  constexpr size_t kBlockSize = 1024;
  double current_sum = 0.0;
  size_t idx = 0;

  while (idx + kBlockSize <= distances.size()) {
    const double partial_sum = Sum(distances.subspan(idx, kBlockSize));
    if (current_sum + partial_sum >= target) break;
    idx += kBlockSize;
    current_sum += partial_sum;
  }

  for (; idx < distances.size(); ++idx) {
    current_sum += distances[idx];
    if (current_sum >= target) return idx;
  }

  return distances.size() - 1;
}

}  // namespace

void GmmUtilsImplInterface::DistancesFromPoint(
    DatapointPtr<double> center, MutableSpan<double> distances) const {
  this->IterateDataset(
      parallelization_pool_,
      [&](size_t offset, const DenseDataset<double>& dataset_batch)
          SCANN_INLINE_LAMBDA {
            auto result_span = distances.subspan(offset, dataset_batch.size());
            if (distance_->specially_optimized_distance_tag() ==
                DistanceMeasure::NOT_SPECIALLY_OPTIMIZED) {
              ParallelFor<1>(
                  IndicesOf(result_span), parallelization_pool_, [&](size_t i) {
                    result_span[i] =
                        distance_->GetDistanceDense(dataset_batch[i], center);
                  });
            } else {
              DenseDistanceOneToMany(*distance_, center, dataset_batch,
                                     result_span, parallelization_pool_);
            }
          });
}

Status GmmUtilsImplInterface::CheckDataDegeneracy() {
  Datapoint<double> centroid;
  SCANN_RETURN_IF_ERROR(this->GetCentroid(&centroid));
  vector<double> distances(this->size());
  this->DistancesFromPoint(centroid.ToPtr(), MakeMutableSpan(distances));
  double d0 = distances[0];
  for (double d : distances) {
    if (d > 0) return OkStatus();

    if (d != d0) return OkStatus();
  }
  return InvalidArgumentError("Data is degenerate: all points are the same!");
}

Status GmmUtilsImplInterface::CheckAllFinite() const {
  Status finite_check_status = OkStatus();
  IterateDataset(
      nullptr, [&finite_check_status](
                   size_t offset, const DenseDataset<double>& dataset_batch) {
        if (!finite_check_status.ok()) return;
        for (size_t i : IndicesOf(dataset_batch)) {
          Status status = VerifyAllFinite(dataset_batch[i].values_span());
          if (!status.ok()) {
            finite_check_status = AnnotateStatus(
                status, StrFormat("(within-batch dp idx = %d)", offset + i));
            break;
          }
        }
      });
  return finite_check_status;
}

class GenericDatasetWithSubset : public GmmUtilsImplInterface {
 public:
  GenericDatasetWithSubset(const Dataset& dataset,
                           ConstSpan<DatapointIndex> subset)
      : dataset_(dataset), subset_(subset) {}

  size_t size() const final { return subset_.size(); }

  size_t dimensionality() const final { return dataset_.dimensionality(); }
  Status GetCentroid(Datapoint<double>* centroid) const final {
    return dataset_.MeanByDimension(subset_, centroid);
  }

  DatapointPtr<double> GetPoint(size_t idx,
                                Datapoint<double>* storage) const final {
    dataset_.GetDenseDatapoint(subset_[idx], storage);
    return storage->ToPtr();
  }

  DatapointPtr<float> GetPoint(size_t idx,
                               Datapoint<float>* storage) const final {
    dataset_.GetDenseDatapoint(subset_[idx], storage);
    return storage->ToPtr();
  }

  DatapointIndex GetOriginalIndex(size_t idx) const final {
    return subset_[idx];
  }

  template <typename DataT, typename CallbackT>
  void IterateDatasetImpl(ThreadPool* parallelization_pool,
                          const CallbackT& callback) const {
    constexpr size_t kBatchSize = 128;
    ParallelFor<1>(
        SeqWithStride<kBatchSize>(subset_.size()), parallelization_pool,
        [&](size_t subset_idx) SCANN_INLINE_LAMBDA {
          const size_t batch_size =
              std::min(kBatchSize, subset_.size() - subset_idx);

          DenseDataset<DataT> dataset_batch;
          dataset_batch.set_dimensionality(dataset_.dimensionality());
          dataset_batch.Reserve(batch_size);
          Datapoint<DataT> dp;
          for (size_t offset : Seq(batch_size)) {
            const DatapointIndex dp_index = subset_[subset_idx + offset];
            dataset_.GetDenseDatapoint(dp_index, &dp);
            CHECK_OK(dataset_batch.Append(dp.ToPtr(), ""));
          }

          callback(subset_idx, dataset_batch);
        });
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallback& callback) const final {
    IterateDatasetImpl<double>(parallelization_pool, callback);
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallbackFloat& callback) const final {
    IterateDatasetImpl<float>(parallelization_pool, callback);
  }

 private:
  const Dataset& dataset_;
  ConstSpan<DatapointIndex> subset_;
};

template <typename T>
class DenseDatasetWrapper : public GmmUtilsImplInterface {
 public:
  explicit DenseDatasetWrapper(const DenseDataset<T>& dataset)
      : dataset_(dataset) {}

  size_t size() const final { return dataset_.size(); }

  size_t dimensionality() const final { return dataset_.dimensionality(); }
  Status GetCentroid(Datapoint<double>* centroid) const final {
    return dataset_.MeanByDimension(centroid);
  }

  DatapointPtr<double> GetPoint(size_t idx,
                                Datapoint<double>* storage) const final {
    return MaybeConvertDatapoint(dataset_[idx], storage);
  }

  DatapointPtr<float> GetPoint(size_t idx,
                               Datapoint<float>* storage) const final {
    return MaybeConvertDatapoint(dataset_[idx], storage);
  }

  DatapointIndex GetOriginalIndex(size_t idx) const final { return idx; }

  template <typename DataT, typename CallbackT>
  void IterateDatasetImpl(ThreadPool* parallelization_pool,
                          const CallbackT& callback) const {
    constexpr size_t kBatchSize = 128;
    ParallelFor<1>(
        SeqWithStride<kBatchSize>(dataset_.size()), parallelization_pool,
        [&](size_t offset) SCANN_INLINE_LAMBDA {
          const size_t batch_size =
              std::min(kBatchSize, dataset_.size() - offset);
          DenseDataset<DataT> dataset_batch;
          dataset_batch.set_dimensionality(dataset_.dimensionality());
          dataset_batch.Reserve(batch_size);
          Datapoint<DataT> storage;
          for (size_t j : Seq(batch_size)) {
            auto dptr = MaybeConvertDatapoint(dataset_[offset + j], &storage);
            CHECK_OK(dataset_batch.Append(dptr, ""));
          }

          callback(offset, dataset_batch);
        });
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallback& callback) const final {
    if constexpr (IsSame<T, double>()) {
      callback(0, reinterpret_cast<const DenseDataset<double>&>(dataset_));
      return;
    }
    IterateDatasetImpl<double>(parallelization_pool, callback);
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallbackFloat& callback) const final {
    if constexpr (IsSame<T, float>()) {
      constexpr size_t kBatchSize = 1024;
      const auto& ds = reinterpret_cast<const DenseDataset<float>&>(dataset_);
      const size_t dims = ds.dimensionality();
      ParallelFor<1>(
          SeqWithStride<kBatchSize>(ds.size()), parallelization_pool,
          [&](size_t offset) SCANN_INLINE_LAMBDA {
            const size_t batch_size = std::min(kBatchSize, ds.size() - offset);
            ConstSpan<float> span(ds.data(offset).data(), batch_size * dims);
            callback(offset, DefaultDenseDatasetView<float>(span, dims));
          });
      return;
    }
    IterateDatasetImpl<float>(parallelization_pool, callback);
  }

 private:
  const DenseDataset<T>& dataset_;
};

template <typename T, bool UseSubset>
class ConstSpanDatasetWrapper : public GmmUtilsImplInterface {
 public:
  explicit ConstSpanDatasetWrapper(ConstSpan<T> dataset,
                                   DimensionIndex dimensionality,
                                   ConstSpan<DatapointIndex> subset)
      : dataset_(dataset),
        dimensionality_(dimensionality),
        size_(UseSubset ? subset.size() : (dataset.size() / dimensionality)),
        subset_(subset) {
    CHECK_EQ(dataset.size() % dimensionality, 0);
  }

  size_t size() const final { return size_; }
  size_t dimensionality() const final { return dimensionality_; }

  Status GetCentroid(Datapoint<double>* centroid) const final {
    centroid->ZeroFill(dimensionality_);
    auto& values = *centroid->mutable_values();
    for (size_t i : Seq(size_)) {
      PointwiseAdd(values.data(), GetDatapointPtr(i).values(), dimensionality_);
    }
    const double multiplier = 1.0 / static_cast<double>(size_);
    for (double& v : values) {
      v *= multiplier;
    }
    return OkStatus();
  }

  DatapointPtr<double> GetPoint(size_t idx,
                                Datapoint<double>* storage) const final {
    auto dptr = GetDatapointPtr(idx);
    return MaybeConvertDatapoint(dptr, storage);
  }

  DatapointPtr<float> GetPoint(size_t idx,
                               Datapoint<float>* storage) const final {
    auto dptr = GetDatapointPtr(idx);
    return MaybeConvertDatapoint(dptr, storage);
  }

  DatapointIndex GetOriginalIndex(size_t idx) const final {
    if constexpr (UseSubset) {
      return subset_[idx];
    }
    return idx;
  }

  template <typename DataT, typename CallbackT>
  void IterateDatasetImpl(ThreadPool* parallelization_pool,
                          const CallbackT& callback) const {
    constexpr size_t kBatchSize = 128;
    ParallelFor<1>(SeqWithStride<kBatchSize>(size_), parallelization_pool,
                   [&](size_t offset) SCANN_INLINE_LAMBDA {
                     const size_t batch_size =
                         std::min(kBatchSize, size_ - offset);
                     DenseDataset<DataT> dataset_batch;
                     dataset_batch.set_dimensionality(dimensionality_);
                     dataset_batch.Reserve(batch_size);
                     Datapoint<DataT> storage;
                     for (size_t j : Seq(batch_size)) {
                       auto dptr1 = GetDatapointPtr(offset + j);
                       auto dptr2 = MaybeConvertDatapoint(dptr1, &storage);
                       CHECK_OK(dataset_batch.Append(dptr2, ""));
                     }

                     callback(offset, dataset_batch);
                   });
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallback& callback) const final {
    IterateDatasetImpl<double>(parallelization_pool, callback);
  }

  void IterateDataset(ThreadPool* parallelization_pool,
                      const IterateDatasetCallbackFloat& callback) const final {
    if constexpr (IsSame<T, float>() && !UseSubset) {
      constexpr size_t kBatchSize = 1024;
      ParallelFor<1>(
          SeqWithStride<kBatchSize>(size_), parallelization_pool,
          [&](size_t offset) SCANN_INLINE_LAMBDA {
            const size_t batch_size = std::min(kBatchSize, size_ - offset);
            auto span = dataset_.subspan(offset * dimensionality_,
                                         batch_size * dimensionality_);
            callback(offset,
                     DefaultDenseDatasetView<float>(span, dimensionality_));
          });
      return;
    }
    IterateDatasetImpl<float>(parallelization_pool, callback);
  }

 private:
  DatapointPtr<T> GetDatapointPtr(size_t idx) const {
    if constexpr (UseSubset) {
      idx = subset_[idx];
    }
    return MakeDatapointPtr(
        dataset_.subspan(idx * dimensionality_, dimensionality_));
  }

  ConstSpan<T> dataset_;
  DatapointIndex dimensionality_;
  DatapointIndex size_;
  ConstSpan<DatapointIndex> subset_;
};

template <typename T>
unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::CreateTyped(
    const DistanceMeasure& distance, const Dataset& dataset,
    ConstSpan<DatapointIndex> subset, ThreadPool* parallelization_pool) {
  DCHECK(subset.empty());
  auto* dense_dataset = dynamic_cast<const DenseDataset<T>*>(&dataset);
  CHECK(dense_dataset);
  return make_unique<DenseDatasetWrapper<T>>(*dense_dataset);
}

unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::Create(
    const DistanceMeasure& distance, const Dataset& dataset,
    ConstSpan<DatapointIndex> subset, ThreadPool* parallelization_pool) {
  unique_ptr<GmmUtilsImplInterface> impl;
  if (dataset.IsDense() && subset.empty()) {
    impl = SCANN_CALL_FUNCTION_BY_TAG(
        dataset.TypeTag(), GmmUtilsImplInterface::CreateTyped, distance,
        dataset, subset, parallelization_pool);
  } else {
    impl = make_unique<GenericDatasetWithSubset>(dataset, subset);
  }
  impl->normalization_ = dataset.normalization();
  impl->distance_ = &distance;
  impl->parallelization_pool_ = parallelization_pool;
  return impl;
}

template <typename T>
unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::Create(
    const DistanceMeasure& distance, ConstSpan<T> data,
    DatapointIndex dimensionality, ConstSpan<DatapointIndex> subset,
    Normalization normalization, ThreadPool* parallelization_pool) {
  unique_ptr<GmmUtilsImplInterface> impl;
  if (subset.empty()) {
    impl = std::make_unique<ConstSpanDatasetWrapper<T, false>>(
        data, dimensionality, subset);
  } else {
    impl = std::make_unique<ConstSpanDatasetWrapper<T, true>>(
        data, dimensionality, subset);
  }
  impl->normalization_ = normalization;
  impl->distance_ = &distance;
  impl->parallelization_pool_ = parallelization_pool;
  return impl;
}

template unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::Create(
    const DistanceMeasure& distance, ConstSpan<float> data,
    DatapointIndex dimensionality, ConstSpan<DatapointIndex> subset,
    Normalization normalization, ThreadPool* parallelization_pool);

template unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::Create(
    const DistanceMeasure& distance, ConstSpan<double> data,
    DatapointIndex dimensionality, ConstSpan<DatapointIndex> subset,
    Normalization normalization, ThreadPool* parallelization_pool);

namespace {

vector<pair<DatapointIndex, double>> UnbalancedPartitionAssignment(
    GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
    const DenseDataset<double>& centers, ThreadPool* pool) {
  vector<pair<DatapointIndex, double>> top1_results(impl->size());

  impl->IterateDataset(
      pool, [&](size_t offset,
                const DenseDataset<double>& dataset_batch) SCANN_INLINE_LAMBDA {
        DCHECK(&dataset_batch);
        DCHECK(&centers);
        DCHECK_GT(dataset_batch.size(), 0);
        DCHECK_GT(centers.size(), 0);
        DCHECK_EQ(centers.dimensionality(), dataset_batch.dimensionality());
        DCHECK(&distance);
        DCHECK(!distance.name().empty());
        DCHECK_OK(VerifyAllFinite(dataset_batch.data()));
        DCHECK_OK(VerifyAllFinite(centers.data()));
        auto results =
            DenseDistanceManyToManyTop1(distance, dataset_batch, centers, pool);
        DCHECK_EQ(results.size(), dataset_batch.size());
        std::copy(results.begin(), results.end(),
                  top1_results.begin() + offset);
      });
  return top1_results;
}

vector<pair<DatapointIndex, double>> UnbalancedFloat32PartitionAssignment(
    GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
    const DenseDataset<double>& centers, ThreadPool* pool) {
  vector<pair<DatapointIndex, double>> top1_results(impl->size());
  DenseDataset<float> centers_fp32;
  centers.ConvertType(&centers_fp32);

  impl->IterateDataset(
      pool,
      [&](size_t offset, DefaultDenseDatasetView<float> dataset_batch)
          SCANN_INLINE_LAMBDA {
            DCHECK(&dataset_batch);
            DCHECK(&centers);
            DCHECK_GT(dataset_batch.size(), 0);
            DCHECK_GT(centers.size(), 0);
            DCHECK_EQ(centers.dimensionality(), dataset_batch.dimensionality());
            DCHECK(&distance);
            DCHECK(!distance.name().empty());
            DCHECK_OK(VerifyAllFinite(dataset_batch.data()));
            DCHECK_OK(VerifyAllFinite(centers.data()));

            auto results = DenseDistanceManyToManyTop1(distance, dataset_batch,
                                                       centers_fp32, nullptr);
            DCHECK_EQ(results.size(), dataset_batch.size());

            std::copy(results.begin(), results.end(),
                      top1_results.begin() + offset);
          });
  return top1_results;
}

}  // namespace

Status GmmUtils::InitializeCenters(const Dataset& dataset,
                                   ConstSpan<DatapointIndex> subset,
                                   int32_t num_clusters,
                                   ConstSpan<float> weights,
                                   DenseDataset<double>* initial_centers) {
  SCANN_RET_CHECK(initial_centers);
  initial_centers->clear();
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  SCANN_RETURN_IF_ERROR(impl->CheckAllFinite())
      << "Non-finite values detected in the initial dataset in "
         "GmmUtils::InitializeCenters.";
  return InitializeCenters(impl.get(), num_clusters, weights, initial_centers);
}

Status GmmUtils::InitializeCenters(GmmUtilsImplInterface* impl,
                                   int32_t num_clusters,
                                   ConstSpan<float> weights,
                                   DenseDataset<double>* initial_centers) {
  switch (opts_.center_initialization_type) {
    case Options::KMEANS_PLUS_PLUS:
      SCANN_RETURN_IF_ERROR(KMeansPPInitializeCenters(
          impl, num_clusters, weights, initial_centers));
      break;
    case Options::RANDOM_INITIALIZATION:
      SCANN_RETURN_IF_ERROR(RandomInitializeCenters(impl, num_clusters, weights,
                                                    initial_centers));
      break;
    case Options::MEAN_DISTANCE_INITIALIZATION:
      SCANN_RETURN_IF_ERROR(MeanDistanceInitializeCenters(
          impl, num_clusters, weights, initial_centers));
      break;
  }
  initial_centers->set_normalization_tag(impl->normalization());
  return OkStatus();
}

Status GmmUtils::MeanDistanceInitializeCenters(
    GmmUtilsImplInterface* impl, int32_t num_clusters, ConstSpan<float> weights,
    DenseDataset<double>* initial_centers) {
  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  centers.set_dimensionality(impl->dimensionality());
  centers.Reserve(num_clusters);

  Datapoint<double> storage;
  SCANN_RETURN_IF_ERROR(impl->GetCentroid(&storage));
  DatapointPtr<double> last_center = storage.ToPtr();

  vector<DatapointIndex> sample_ids;
  vector<double> distance_weights(dataset_size, 0.0);

  impl->DistancesFromPoint(last_center, MakeMutableSpan(distance_weights));
  if (!weights.empty()) {
    SCANN_RET_CHECK_EQ(weights.size(), distance_weights.size());
    for (size_t i : IndicesOf(weights)) {
      distance_weights[i] *= weights[i];
    }
  }
  SCANN_RETURN_IF_ERROR(VerifyAllFinite(last_center.values_span()))
      << "(Center Number = " << centers.size() << ")";
  double min_dist = 0.0;
  double sum = 0.0;
  for (size_t j : Seq(distance_weights.size())) {
    SCANN_RET_CHECK(!std::isnan(distance_weights[j]))
        << "NaN distance_weights found (j = " << j << ").";
    SCANN_RET_CHECK(std::isfinite(distance_weights[j]))
        << "Infinite distance_weights found (j = " << j << ").";
    min_dist = std::min(min_dist, distance_weights[j]);
    sum += distance_weights[j];
  }

  if (min_dist < 0.0) {
    VLOG(1) << "Biasing to get rid of negative distance_weights. (min_dist = "
            << min_dist << ")";
    BiasDistances(-min_dist, MakeMutableSpan(distance_weights));
    sum += -min_dist * distance_weights.size();
  }

  while (centers.size() < num_clusters) {
    for (DatapointIndex idx : sample_ids) {
      sum -= distance_weights[idx];
      distance_weights[idx] = 0.0;
    }

    SCANN_RET_CHECK(!std::isnan(sum)) << "NaN distance_weights sum found.";
    SCANN_RET_CHECK(std::isfinite(sum))
        << "Infinite distance_weights sum found.";
    DatapointIndex sample_id = GetSample(&random_, distance_weights, sum, true);
    sample_ids.push_back(sample_id);
    last_center = impl->GetPoint(sample_id, &storage);
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(storage.values()));
    centers.AppendOrDie(last_center, "");
  }

  *initial_centers = std::move(centers);
  return OkStatus();
}

Status GmmUtils::KMeansPPInitializeCenters(
    GmmUtilsImplInterface* impl, int32_t num_clusters, ConstSpan<float> weights,
    DenseDataset<double>* initial_centers) {
  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  centers.set_dimensionality(impl->dimensionality());
  centers.Reserve(num_clusters);

  Datapoint<double> storage;
  SCANN_RETURN_IF_ERROR(impl->GetCentroid(&storage));
  DatapointPtr<double> last_center = storage.ToPtr();

  vector<DatapointIndex> sample_ids;
  vector<double> distances(dataset_size, 0.0);
  vector<double> temp(dataset_size);
  while (centers.size() < num_clusters) {
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(last_center.values_span()))
        << "(Center Number = " << centers.size() << ")";
    impl->DistancesFromPoint(last_center, MakeMutableSpan(temp));
    double min_dist = 0.0;
    double sum = 0.0;
    for (size_t j : Seq(distances.size())) {
      SCANN_RET_CHECK(!std::isnan(temp[j]))
          << "NaN distances found (j = " << j << ").";
      SCANN_RET_CHECK(std::isfinite(temp[j]))
          << "Infinite distances found (j = " << j << ").";
      distances[j] += temp[j];
      min_dist = std::min(min_dist, distances[j]);
      sum += distances[j];
    }

    if (min_dist < 0.0) {
      VLOG(1) << "Biasing to get rid of negative distances. (min_dist = "
              << min_dist << ")";
      BiasDistances(-min_dist, MakeMutableSpan(distances));
      sum += -min_dist * distances.size();
    }

    for (DatapointIndex idx : sample_ids) {
      sum -= distances[idx];
      distances[idx] = 0.0;
    }

    SCANN_RET_CHECK(!std::isnan(sum)) << "NaN distances sum found.";
    SCANN_RET_CHECK(std::isfinite(sum)) << "Infinite distances sum found.";
    const DatapointIndex sample_id = [&] {
      const bool is_first = centers.empty();
      if (weights.empty()) {
        return GetSample(&random_, distances, sum, is_first);
      } else {
        vector<double> distance_weights = distances;
        double weighted_sum = 0.0;
        for (size_t weight_idx : IndicesOf(weights)) {
          distance_weights[weight_idx] *= weights[weight_idx];
          weighted_sum += distance_weights[weight_idx];
        }
        return GetSample(&random_, distance_weights, weighted_sum, is_first);
      }
    }();
    sample_ids.push_back(sample_id);
    last_center = impl->GetPoint(sample_id, &storage);
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(storage.values()));

    uint32_t big_endian_sample_id = absl::ghtonl(sample_id);
    std::string docid;
    docid.resize(sizeof(big_endian_sample_id));
    memcpy(docid.data(), &big_endian_sample_id, sizeof(big_endian_sample_id));
    centers.AppendOrDie(last_center, docid);
  }

  *initial_centers = std::move(centers);
  return OkStatus();
}

Status GmmUtils::RandomInitializeCenters(
    GmmUtilsImplInterface* impl, int32_t num_clusters, ConstSpan<float> weights,
    DenseDataset<double>* initial_centers) {
  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  Datapoint<double> storage;
  centers.set_dimensionality(impl->dimensionality());
  centers.Reserve(num_clusters);
  absl::flat_hash_set<DatapointIndex> center_ids;
  if (weights.empty()) {
    while (center_ids.size() < num_clusters) {
      center_ids.insert(
          absl::Uniform<DatapointIndex>(random_, 0, dataset_size));
    }
  } else {
    absl::discrete_distribution<size_t> distrib(weights.begin(), weights.end());
    while (center_ids.size() < num_clusters) {
      center_ids.insert(distrib(random_));
    }
  }
  for (DatapointIndex idx : center_ids) {
    SCANN_RETURN_IF_ERROR(centers.Append(impl->GetPoint(idx, &storage)));
  }

  *initial_centers = std::move(centers);
  return OkStatus();
}

namespace {

bool IsStdIota(ConstSpan<DatapointIndex> indices) {
  for (size_t j : Seq(indices.size())) {
    if (indices[j] != j) return false;
  }
  return true;
}

}  // namespace

__attribute__((no_sanitize("float-divide-by-zero"))) SCANN_OUTLINE Status
GmmUtils::ComputeKmeansClustering(
    const Dataset& dataset, const int32_t num_clusters,
    DenseDataset<double>* final_centers,
    const ComputeKmeansClusteringOptions& kmeans_opts) {
  ConstSpan<DatapointIndex> subset = kmeans_opts.subset;
  if (dataset.IsDense() && subset.size() == dataset.size() &&
      IsStdIota(subset)) {
    subset = ConstSpan<DatapointIndex>();
  }
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  return ComputeKmeansClustering(impl.get(), num_clusters, final_centers,
                                 kmeans_opts);
}

__attribute__((no_sanitize("float-divide-by-zero"))) SCANN_OUTLINE Status
GmmUtils::ComputeKmeansClustering(
    GmmUtilsImplInterface* impl, const int32_t num_clusters,
    DenseDataset<double>* final_centers,
    const ComputeKmeansClusteringOptions& kmeans_opts) {
  static_assert(
      std::numeric_limits<double>::is_iec559,
      "Function depends on IEEE divide-by-zero semantics for correctness");

  SCANN_RET_CHECK(final_centers);
  const bool spherical = kmeans_opts.spherical;
  DenseDataset<double> centers;
  SCANN_RETURN_IF_ERROR(
      InitializeCenters(impl, num_clusters, kmeans_opts.weights, &centers));
  SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
      << "Initial centers contain NaN/infinity.";
  if (kmeans_opts.spherical) {
    if (impl->normalization() != UNITL2NORM) {
      return InvalidArgumentError("Input vectors must be unit L2-norm.");
    }
    if (centers.normalization() != UNITL2NORM) {
      return InvalidArgumentError("Initial centers must be unit L2-norm.");
    }
  }

  if (opts_.max_iterations <= 0) {
    return InvalidArgumentError(
        "Zero or negative iterations specified in GmmUtils::Options!");
  }
  if (opts_.min_cluster_size <= 0) {
    return InvalidArgumentError(
        "Zero or negative min cluster size specified in "
        "GmmUtils::Options!");
  }
  const size_t dataset_size = impl->size();
  if (!num_clusters) {
    return InvalidArgumentError("Initial centers are undefined.");
  }
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters);
  }

  const size_t min_cluster_size =
      std::min<size_t>(opts_.min_cluster_size, dataset_size / num_clusters);
  SCANN_RET_CHECK_GE(min_cluster_size, 1);

  if (!impl->CheckDataDegeneracy().ok()) {
    LOG_EVERY_N(WARNING, 1000000) << StrFormat(
        "All %d points are exactly the same in the partition.", dataset_size);
  }

  vector<double> old_means(num_clusters, -1.0);
  vector<double> new_means(num_clusters, -1.0);

  vector<uint32_t> partition_sizes(num_clusters);
  vector<pair<DatapointIndex, double>> top1_results;

  ThreadPool* pool = opts_.parallelization_pool.get();
  const absl::Time deadline = absl::Now() + opts_.max_iteration_duration;
  for (size_t iteration : Seq(opts_.max_iterations + 1)) {
    auto centers_view = DefaultDenseDatasetView<double>(centers);
    switch (opts_.partition_assignment_type) {
      case GmmUtils::Options::UNBALANCED:
        top1_results =
            UnbalancedPartitionAssignment(impl, *distance_, centers, pool);
        break;
      case GmmUtils::Options::UNBALANCED_FLOAT32:
        top1_results = UnbalancedFloat32PartitionAssignment(impl, *distance_,
                                                            centers, pool);
        break;
      case GmmUtils::Options::GREEDY_BALANCED:
      case GmmUtils::Options::MIN_COST_MAX_FLOW:
        LOG(ERROR) << "Unsupported partition_assignment_type.";
        break;
      default:
        LOG(FATAL) << "Invalid partition assignment type.";
    }
    QCHECK_EQ(top1_results.size(), dataset_size);

    std::swap(old_means, new_means);
    std::fill(new_means.begin(), new_means.end(), 0.0);
    std::fill(partition_sizes.begin(), partition_sizes.end(), 0);

    for (size_t j : Seq(dataset_size)) {
      const uint32_t cluster_idx = top1_results[j].first;
      QCHECK_LT(cluster_idx, num_clusters);
      const double distance = top1_results[j].second;
      QCHECK(std::isfinite(distance));
      partition_sizes[cluster_idx] += 1;
      new_means[cluster_idx] += distance;
    }
    for (size_t c : Seq(num_clusters)) {
      new_means[c] /= partition_sizes[c];
    }

    bool converged = true;
    for (size_t c : Seq(num_clusters)) {
      const double delta = new_means[c] - old_means[c];
      if (fabs(delta) > opts_.epsilon ||
          partition_sizes[c] < min_cluster_size) {
        converged = false;
        break;
      }
    }
    if (converged) {
      VLOG(1) << StrFormat("Converged in %d iterations.", iteration);
      break;
    }
    if (iteration == opts_.max_iterations || absl::Now() > deadline) {
      VLOG(1) << StrFormat("Exiting without converging after %d iterations.",
                           iteration);
      break;
    }

    if (kmeans_opts.weights.empty()) {
      SCANN_RETURN_IF_ERROR(
          RecomputeCentroidsSimple(top1_results, impl, partition_sizes,
                                   spherical, &centers_view, &new_means));
    } else {
      SCANN_RETURN_IF_ERROR(RecomputeCentroidsWeighted(
          top1_results, impl, partition_sizes, kmeans_opts.weights, spherical,
          &centers_view, &new_means));
    }
  }

  if (kmeans_opts.final_partitions) {
    vector<vector<DatapointIndex>> partitions(num_clusters);
    for (size_t c : Seq(num_clusters)) {
      partitions[c].reserve(partition_sizes[c]);
    }

    for (size_t j : Seq(dataset_size)) {
      const uint32_t cluster_idx = top1_results[j].first;
      const DatapointIndex dp_idx = impl->GetOriginalIndex(j);
      partitions[cluster_idx].push_back(dp_idx);
    }
    *kmeans_opts.final_partitions = std::move(partitions);
  }

  *final_centers = std::move(centers);
  return OkStatus();
}

StatusOr<double> GmmUtils::ComputeSpillingThreshold(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const DenseDataset<double>& centers,
    const DatabaseSpillingConfig::SpillingType spilling_type,
    const float total_spill_factor, const DatapointIndex max_centers) {
  if (max_centers <= 1) {
    return InvalidArgumentError(
        "max_centers must be > 1 for ComputeSpillingThreshold.");
  }

  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  const size_t dataset_size = impl->size();

  if (centers.size() < 2) {
    return InvalidArgumentError(
        "Need at least two centers for ComputeSpillingThreshold.");
  }

  if (dataset_size == 0) {
    return InvalidArgumentError("The input dataset is empty.");
  }

  if (spilling_type != DatabaseSpillingConfig::ADDITIVE &&
      spilling_type != DatabaseSpillingConfig::MULTIPLICATIVE) {
    SCANN_RET_CHECK_EQ(spilling_type, DatabaseSpillingConfig::NO_SPILLING)
        << "Invalid spilling type.";
    return NAN;
  }

  if (total_spill_factor <= 1) {
    return 0.0;
  }

  const DatapointIndex max_neighbors =
      std::min<DatapointIndex>(centers.size(), max_centers);

  vector<double> spills;

  Datapoint<double> storage;
  vector<double> distances(centers.size());
  for (size_t j : Seq(dataset_size)) {
    DenseDistanceOneToMany(*distance_, impl->GetPoint(j, &storage), centers,
                           MakeMutableSpan(distances),
                           opts_.parallelization_pool.get());
    TopNAmortizedConstant<double, std::less<double>> top_n(max_neighbors);
    top_n.reserve(max_neighbors + 1);
    for (double dist : distances) {
      top_n.push(dist);
    }
    auto top_items = top_n.Take();

    if (spilling_type == DatabaseSpillingConfig::ADDITIVE) {
      for (DatapointIndex k : Seq(1, max_neighbors)) {
        spills.push_back(top_items[k] - top_items[0]);
      }
    } else {
      DCHECK_EQ(spilling_type, DatabaseSpillingConfig::MULTIPLICATIVE);

      for (DatapointIndex k : Seq(1, max_neighbors)) {
        if (top_items[0] == 0.0 && top_items[k] == 0.0) {
          return InternalError(
              "Duplicate centers.  Cannot compute spilling threshold.");
        }
        if (top_items[0] == 0.0) {
          spills.push_back(1.0);
        } else {
          spills.push_back(top_items[k] / top_items[0]);
        }
      }
    }
  }

  int32_t threshold_index;

  DCHECK_GT(spills.size(), 0);
  if (max_neighbors <= total_spill_factor) {
    if (max_neighbors == total_spill_factor) {
      LOG(WARNING) << "max_neighbors == total_spill_factor";
    }
    return *std::max_element(spills.begin(), spills.end());
  } else {
    threshold_index = std::floor((total_spill_factor - 1) * dataset_size);
    DCHECK_GT(spills.size(), threshold_index);
    std::nth_element(spills.begin(), spills.begin() + threshold_index,
                     spills.end());
    return spills[threshold_index];
  }
}

namespace {
template <typename FloatT>
void NormalizeCentroid(MutableSpan<FloatT> mut_centroid, double divisor) {
  if (divisor == 0) {
    LOG_FIRST_N(WARNING, 1) << "Could not normalize centroid due to zero "
                               "norm or empty or zero-weight partition.";
    return;
  }
  const FloatT multiplier = 1.0 / divisor;
  for (FloatT& dim : mut_centroid) {
    dim *= multiplier;
  }
}
}  // namespace

template <typename FloatT>
Status GmmUtils::RecomputeCentroidsSimple(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) {
  enum : size_t { kParallelAggregate = 4 };
  const size_t dataset_size = impl->size();
  const size_t dimensionality = impl->dimensionality();
  for (size_t i : IndicesOf(*centroids)) {
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(i)), dimensionality);
    std::fill(mut_centroid.begin(), mut_centroid.end(), 0.0);
  }
  if (impl->GetThreadPool() &&
      top1_results.size() >= centroids->size() * kParallelAggregate * 2) {
    size_t num_centroids = centroids->size();
    size_t num_tmp_centroids = num_centroids * kParallelAggregate;
    DenseDataset<FloatT> tmp_centroids(
        std::vector<FloatT>(kParallelAggregate * dimensionality * num_centroids,
                            0.0f),
        num_tmp_centroids);

    size_t db_slice_size =
        (top1_results.size() + kParallelAggregate - 1) / kParallelAggregate;
    ParallelFor<1>(
        Seq(kParallelAggregate), impl->GetThreadPool(), [&](size_t thread_id) {
          size_t db_slice_start = thread_id * db_slice_size;
          size_t db_slice_end =
              std::min((thread_id + 1) * db_slice_size, top1_results.size());
          Datapoint<FloatT> storage;
          for (auto i = db_slice_start; i < db_slice_end; ++i) {
            size_t cluster_idx = top1_results[i].first;
            size_t tmp_centroid_idx = thread_id * num_centroids + cluster_idx;
            QCHECK_LT(tmp_centroid_idx, tmp_centroids.size());
            QCHECK_LT(i, impl->size());
            auto centroid_span = tmp_centroids.mutable_data(tmp_centroid_idx);
            auto datapoint = impl->GetPoint(i, &storage).values_span();
            QCHECK_EQ(centroid_span.size(), datapoint.size());
            for (size_t j : Seq(dimensionality)) {
              centroid_span[j] += datapoint[j];
            }
          }
        });
    for (size_t i : IndicesOf(*centroids)) {
      MutableSpan<FloatT> out_centroid = MakeMutableSpan(
          const_cast<FloatT*>(centroids->GetPtr(i)), dimensionality);
      for (size_t j = 0; j < num_tmp_centroids; j += num_centroids) {
        auto tmp_centroid = tmp_centroids.data(i + j);
        for (size_t jj : Seq(dimensionality)) {
          out_centroid[jj] += tmp_centroid[jj];
        }
      }
    }
  } else {
    Datapoint<FloatT> storage;
    for (size_t j : Seq(dataset_size)) {
      const uint32_t cluster_idx = top1_results[j].first;
      MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
          const_cast<FloatT*>(centroids->GetPtr(cluster_idx)), dimensionality);
      ConstSpan<FloatT> datapoint = impl->GetPoint(j, &storage).values_span();
      SCANN_RET_CHECK_EQ(mut_centroid.size(), dimensionality);
      SCANN_RET_CHECK_EQ(datapoint.size(), dimensionality);
      for (size_t jj : Seq(dimensionality)) {
        mut_centroid[jj] += datapoint[jj];
      }
    }
  }

  for (size_t c : IndicesOf(*centroids)) {
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(c)), dimensionality);
    auto centroid_dptr = MakeDatapointPtr<FloatT>(mut_centroid);
    const double divisor = spherical ? std::sqrt(SquaredL2Norm(centroid_dptr))
                                     : static_cast<double>(partition_sizes[c]);
    NormalizeCentroid(mut_centroid, divisor);
  }
  SCANN_RETURN_IF_ERROR(VerifyDatasetAllFinite(*centroids));
  return ReinitializeCenters(top1_results, impl, partition_sizes, spherical,
                             centroids, convergence_means);
  return OkStatus();
}

template Status GmmUtils::RecomputeCentroidsSimple(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<float>* centroids,
    std::vector<double>* convergence_means);
template Status GmmUtils::RecomputeCentroidsSimple(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<double>* centroids,
    std::vector<double>* convergence_means);

template <typename FloatT>
Status GmmUtils::RecomputeCentroidsWeighted(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    ConstSpan<float> weights, bool spherical,
    DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) {
  const size_t dataset_size = impl->size();
  SCANN_RET_CHECK_EQ(weights.size(), dataset_size);
  const size_t dimensionality = impl->dimensionality();
  for (size_t i : IndicesOf(*centroids)) {
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(i)), dimensionality);
    std::fill(mut_centroid.begin(), mut_centroid.end(), 0.0);
  }
  Datapoint<FloatT> storage;
  vector<FloatT> denominators(centroids->size());
  for (size_t dp_idx : Seq(dataset_size)) {
    const size_t cluster_idx = top1_results[dp_idx].first;
    const FloatT weight = weights[dp_idx];
    denominators[cluster_idx] += weight;
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(cluster_idx)), dimensionality);
    ConstSpan<FloatT> datapoint =
        impl->GetPoint(dp_idx, &storage).values_span();
    SCANN_RET_CHECK_EQ(mut_centroid.size(), dimensionality);
    SCANN_RET_CHECK_EQ(datapoint.size(), dimensionality);
    for (size_t dim_idx : Seq(dimensionality)) {
      mut_centroid[dim_idx] += datapoint[dim_idx] * weight;
    }
  }

  for (size_t c : IndicesOf(*centroids)) {
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(c)), dimensionality);
    auto centroid_dptr = MakeDatapointPtr<FloatT>(mut_centroid);
    const FloatT divisor =
        spherical ? std::sqrt(SquaredL2Norm(centroid_dptr)) : denominators[c];
    NormalizeCentroid(mut_centroid, divisor);
  }
  SCANN_RETURN_IF_ERROR(VerifyDatasetAllFinite(*centroids));
  return ReinitializeCenters(top1_results, impl, partition_sizes, spherical,
                             centroids, convergence_means);
}

template Status GmmUtils::RecomputeCentroidsWeighted(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    ConstSpan<float> weights, bool spherical,
    DenseDatasetView<float>* centroids, std::vector<double>* convergence_means);

template Status GmmUtils::RecomputeCentroidsWeighted(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    ConstSpan<float> weights, bool spherical,
    DenseDatasetView<double>* centroids,
    std::vector<double>* convergence_means);

template <typename FloatT>
Status GmmUtils::ReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) {
  switch (opts_.center_reassignment_type) {
    case Options::RANDOM_REASSIGNMENT:
      SCANN_RETURN_IF_ERROR(
          RandomReinitializeCenters(top1_results, impl, partition_sizes,
                                    spherical, centroids, convergence_means));
      SCANN_RETURN_IF_ERROR(VerifyDatasetAllFinite(*centroids))
          << "RandomReinitializeCenters";
      break;
    case Options::SPLIT_LARGEST_CLUSTERS:
      SCANN_RETURN_IF_ERROR(SplitLargeClusterReinitialization(
          partition_sizes, spherical, centroids, convergence_means));
      SCANN_RETURN_IF_ERROR(VerifyDatasetAllFinite(*centroids))
          << "SplitLargeClusterReinitialization";
      break;
    case Options::PCA_SPLITTING:
      SCANN_RETURN_IF_ERROR(
          PCAKmeansReinitialization(top1_results, impl, partition_sizes,
                                    spherical, centroids, convergence_means));
      SCANN_RETURN_IF_ERROR(VerifyDatasetAllFinite(*centroids))
          << "PCAKmeansReinitialization";
      break;
  }
  return OkStatus();
}

template Status GmmUtils::ReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<float>* centroids,
    std::vector<double>* convergence_means);

template Status GmmUtils::ReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<double>* centroids,
    std::vector<double>* convergence_means);

template <typename FloatT>
Status GmmUtils::RandomReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) {
  Datapoint<FloatT> storage;
  int num_reinit_this_iter = 0;
  const uint32_t dimensionality = centroids->dimensionality();
  const size_t dataset_size = impl->size();

  const size_t min_cluster_size = std::min<size_t>(
      opts_.min_cluster_size, dataset_size / partition_sizes.size());
  SCANN_RET_CHECK_GE(min_cluster_size, 1);

  for (size_t c : Seq(centroids->size())) {
    if (partition_sizes[c] >= min_cluster_size) continue;

    num_reinit_this_iter++;
    (*convergence_means)[c] = -1.0;

    DatapointIndex rand_idx = 0;
    uint32_t cluster_idx = 0;
    do {
      rand_idx = absl::Uniform<DatapointIndex>(random_, 0, dataset_size);
      cluster_idx = top1_results[rand_idx].first;
    } while (partition_sizes[cluster_idx] < min_cluster_size);

    ConstSpan<FloatT> rand_point =
        impl->GetPoint(rand_idx, &storage).values_span();
    ConstSpan<FloatT> old_center =
        MakeConstSpan(centroids->GetPtr(cluster_idx), dimensionality);
    MutableSpan<FloatT> mut_centroid = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(c)), dimensionality);
    SCANN_RET_CHECK_EQ(rand_point.size(), dimensionality);
    for (size_t jj : Seq(dimensionality)) {
      mut_centroid[jj] = old_center[jj] +
                         opts_.perturbation * (rand_point[jj] - old_center[jj]);
    }

    if (spherical) {
      const double norm =
          std::sqrt(SquaredL2Norm(MakeDatapointPtr<FloatT>(mut_centroid)));
      if (norm == 0) {
        LOG_FIRST_N(WARNING, 1)
            << "Could not normalize centroid due to zero norm.";
        continue;
      }
      const double multiplier = 1.0 / norm;
      for (size_t jj : Seq(dimensionality)) {
        mut_centroid[jj] *= multiplier;
      }
    }
  }
  VLOG(1) << StrFormat("Reinitialized %d small clusters.",
                       num_reinit_this_iter);
  return OkStatus();
}

template Status GmmUtils::RandomReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<float>* centroids,
    std::vector<double>* convergence_means);

template Status GmmUtils::RandomReinitializeCenters(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<double>* centroids,
    std::vector<double>* convergence_means);

template <typename FloatT>
Status GmmUtils::SplitLargeClusterReinitialization(
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) {
  std::vector<uint32_t> sorted_partition_sizes(partition_sizes.size());
  std::vector<uint32_t> partition_permutation(partition_sizes.size());
  std::copy(partition_sizes.begin(), partition_sizes.end(),
            sorted_partition_sizes.begin());
  std::iota(partition_permutation.begin(), partition_permutation.end(), 0);

  ZipSortBranchOptimized(
      std::greater<uint32_t>(), sorted_partition_sizes.begin(),
      sorted_partition_sizes.end(), partition_permutation.begin(),
      partition_permutation.end());

  Datapoint<double> storage;
  int num_reinit_this_iter = 0;
  const uint32_t dim = centroids->dimensionality();
  const double perturbation_factor =
      std::max(opts_.perturbation, DBL_EPSILON * dim);
  const size_t dimensionality = centroids->dimensionality();

  for (size_t big_cluster_idx : Seq(centroids->size())) {
    if (sorted_partition_sizes[big_cluster_idx] < opts_.max_cluster_size) break;
    num_reinit_this_iter++;

    size_t small_cluster_idx = centroids->size() - 1 - big_cluster_idx;
    (*convergence_means)[small_cluster_idx] = -1.0;

    if (small_cluster_idx <= big_cluster_idx) break;

    Eigen::VectorXd rand_direction(dim);
    for (const auto& i : Seq(dim)) {
      rand_direction[i] = absl::Gaussian<double>(random_);
    }
    rand_direction.normalize();
    rand_direction *= perturbation_factor;
    MutableSpan<FloatT> big_cluster = MakeMutableSpan(
        const_cast<FloatT*>(
            centroids->GetPtr(partition_permutation[big_cluster_idx])),
        dimensionality);
    MutableSpan<FloatT> small_cluster = MakeMutableSpan(
        const_cast<FloatT*>(
            centroids->GetPtr(partition_permutation[small_cluster_idx])),
        dimensionality);
    for (const auto& d : Seq(dim)) {
      small_cluster[d] = big_cluster[d] + rand_direction[d];
      big_cluster[d] = big_cluster[d] - rand_direction[d];
    }
    if (spherical) {
      FloatT big_cluster_norm =
          1.0f / sqrt(SquaredL2Norm(MakeDatapointPtr<FloatT>(big_cluster)));
      FloatT small_cluster_norm =
          1.0f / sqrt(SquaredL2Norm(MakeDatapointPtr<FloatT>(small_cluster)));
      for (size_t d : Seq(dim)) {
        big_cluster[d] *= big_cluster_norm;
        small_cluster[d] *= small_cluster_norm;
      }
    }
  }
  if (num_reinit_this_iter) {
    LOG(INFO) << StrFormat("Reinitialized %d clusters.", num_reinit_this_iter);
  }
  return OkStatus();
}

template <typename FloatT>
Status GmmUtils::PCAKmeansReinitialization(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<FloatT>* centroids,
    std::vector<double>* convergence_means) const {
  using Eigen::aligned_allocator;
  using Eigen::Map;
  using Eigen::SelfAdjointEigenSolver;
  using Matrix = Eigen::Matrix<FloatT, Eigen::Dynamic, Eigen::Dynamic>;
  using RowVector = Eigen::RowVector<FloatT, Eigen::Dynamic>;
  using RowVectorD = Eigen::RowVector<double, Eigen::Dynamic>;
  using Vector = Eigen::Vector<FloatT, Eigen::Dynamic>;

  uint32_t dim = centroids->dimensionality();
  std::vector<uint32_t> sorted_partition_sizes(partition_sizes.size());
  std::vector<uint32_t> partition_permutation(partition_sizes.size());
  std::copy(partition_sizes.begin(), partition_sizes.end(),
            sorted_partition_sizes.begin());
  std::iota(partition_permutation.begin(), partition_permutation.end(), 0);

  ZipSortBranchOptimized(
      std::greater<uint32_t>(), sorted_partition_sizes.begin(),
      sorted_partition_sizes.end(), partition_permutation.begin(),
      partition_permutation.end());

  absl::flat_hash_map<uint32_t, uint32_t> clusters_to_split;
  for (size_t sorted_partition_idx : Seq(sorted_partition_sizes.size())) {
    if (sorted_partition_sizes[sorted_partition_idx] < opts_.max_cluster_size)
      break;
    clusters_to_split.emplace(partition_permutation[sorted_partition_idx],
                              sorted_partition_idx);
  }
  if (!clusters_to_split.empty()) {
    LOG(INFO) << "Going to compute " << clusters_to_split.size()
              << " covariances for cluster splitting";
  }
  absl::Time cov_start = absl::Now();

  std::vector<Matrix, aligned_allocator<Matrix>> covariances(
      clusters_to_split.size());

  std::vector<Vector, aligned_allocator<Vector>> normalized_centroids(
      clusters_to_split.size());

  for (const auto& i : Seq(covariances.size())) {
    covariances[i].setZero(dim, dim);
    if (spherical) {
      Map<const RowVector> centroid(centroids->GetPtr(partition_permutation[i]),
                                    dim);
      normalized_centroids[i] = centroid.normalized();
    }
  }

  Datapoint<double> storage;
  for (const auto& i : Seq(top1_results.size())) {
    const uint32_t cluster_idx = top1_results[i].first;
    auto it = clusters_to_split.find(cluster_idx);

    if (it == clusters_to_split.end()) continue;

    const uint32_t sorted_partition_idx = it->second;
    if (!sorted_partition_idx) continue;
    SCANN_RET_CHECK_EQ(cluster_idx,
                       partition_permutation[sorted_partition_idx]);

    Map<const RowVector> centroid(centroids->GetPtr(cluster_idx), dim);
    impl->GetPoint(i, &storage);
    Map<const RowVectorD> dp(storage.values().data(), dim);
    Vector x = dp.cast<FloatT>() - centroid;
    if (spherical) {
      x = x - x.dot(normalized_centroids[sorted_partition_idx]) *
                  normalized_centroids[sorted_partition_idx];
    }
    covariances[sorted_partition_idx] += x * x.transpose();
  }

  const uint32_t avg_size = impl->size() / centroids->size();
  uint32_t min_partition_idx = sorted_partition_sizes.size();
  for (const auto& i : Seq(covariances.size())) {
    const uint32_t multiple_of_avg = (sorted_partition_sizes[i] - 1) / avg_size;
    const uint32_t num_split_directions = std::min(
        opts_.max_power_of_2_split, 32 - absl::countl_zero(multiple_of_avg));

    covariances[i] /= sorted_partition_sizes[i];
    SelfAdjointEigenSolver<Matrix> eig_solver;
    eig_solver.compute(covariances[i]);
    std::vector<Vector, aligned_allocator<Vector>> split_directions;

    for (int j = dim - 1;
         j >= 0 && split_directions.size() < num_split_directions; --j) {
      const FloatT stdev = std::sqrt(eig_solver.eigenvalues()[j]);
      const FloatT scaling_factor =
          std::max(stdev * static_cast<FloatT>(opts_.perturbation),
                   std::numeric_limits<FloatT>::epsilon() * dim);
      split_directions.push_back(eig_solver.eigenvectors().col(j) *
                                 scaling_factor);
    }

    const uint64_t combinatoric_limit = 1ULL << split_directions.size();
    Vector old_centroid(dim);
    Vector centroid_storage(dim);
    auto old_centroid_span = MakeMutableSpan(
        const_cast<FloatT*>(centroids->GetPtr(partition_permutation[i])), dim);
    std::copy(old_centroid_span.begin(), old_centroid_span.end(),
              old_centroid.begin());
    centroid_storage = old_centroid;
    for (const auto& k : Seq(split_directions.size())) {
      centroid_storage -= split_directions[k];
    }
    if (spherical) {
      centroid_storage.normalize();
    }
    centroid_storage.eval();

    std::copy(centroid_storage.begin(), centroid_storage.end(),
              old_centroid_span.begin());

    for (uint64_t j = 1; j < combinatoric_limit; ++j) {
      const uint32_t small_cluster_index =
          partition_permutation[--min_partition_idx];
      if (min_partition_idx <= i) {
        goto end_covariance_loop;
      }
      (*convergence_means)[small_cluster_index] = -1.0;
      auto override_centroid = MakeMutableSpan(
          const_cast<FloatT*>(centroids->GetPtr(small_cluster_index)), dim);

      centroid_storage = old_centroid;
      for (const auto& k : Seq(split_directions.size())) {
        const double sign = j & (1ULL << k) ? 1.0 : -1.0;
        centroid_storage += sign * split_directions[k];
      }

      centroid_storage.eval();
      std::copy(centroid_storage.begin(), centroid_storage.end(),
                override_centroid.begin());
    }
  }
end_covariance_loop:
  if (spherical) {
    for (size_t c : IndicesOf(*centroids)) {
      MutableSpan<FloatT> mut_centroid =
          MakeMutableSpan(const_cast<FloatT*>(centroids->GetPtr(c)), dim);
      auto centroid_dptr = MakeDatapointPtr<FloatT>(mut_centroid);
      const double divisor = std::sqrt(SquaredL2Norm(centroid_dptr));
      NormalizeCentroid(mut_centroid, divisor);
    }
  }
  absl::Time cov_end = absl::Now();
  if (!clusters_to_split.empty()) {
    LOG(INFO) << "Spent " << absl::ToDoubleSeconds(cov_end - cov_start)
              << " seconds for computing covariances";
  }
  return OkStatus();
}

template Status GmmUtils::PCAKmeansReinitialization(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<float>* centroids,
    std::vector<double>* convergence_means) const;

template Status GmmUtils::PCAKmeansReinitialization(
    ConstSpan<pair<DatapointIndex, double>> top1_results,
    GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
    bool spherical, DenseDatasetView<double>* centroids,
    std::vector<double>* convergence_means) const;

}  // namespace research_scann
