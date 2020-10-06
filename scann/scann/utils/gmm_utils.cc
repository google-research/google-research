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

#include "scann/utils/gmm_utils.h"

#include <cfloat>
#include <limits>

#include "Eigen/Dense"
#include "Eigen/StdVector"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Core/util/Memory.h"
#include "Eigen/src/SVD/JacobiSVD.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/distributions.h"
#include "absl/time/time.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/docid_collection.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

#include "scann/oss_wrappers/scann_status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/top_n.h"

namespace tensorflow {
namespace scann_ops {

GmmUtils::GmmUtils(shared_ptr<const DistanceMeasure> distance, Options opts)
    : distance_(std::move(distance)), opts_(opts), random_(opts_.seed) {}

namespace {

void BiasDistances(double bias, MutableSpan<double> distances) {
  if (bias == 0.0) return;
  for (size_t j : Seq(distances.size())) {
    distances[j] += bias;
  }
}

void OffsetNegativeDistances(MutableSpan<double> distances) {
  double min_dist = 0.0;
  for (double d : distances) {
    min_dist = std::min(min_dist, d);
  }
  if (min_dist >= 0.0) return;
  const double bias = -min_dist;
  for (size_t j : Seq(distances.size())) {
    distances[j] += bias;
  }
}

DatapointIndex GetSample(MTRandom* random, ConstSpan<double> distances,
                         double distances_sum, bool is_first) {
  if (distances_sum <= 0.0 || std::isnan(distances_sum)) {
    VLOG(1) << StrFormat(
        "All %d points are zero distance from the centers (distances_sum = "
        "%f).",
        distances.size(), distances_sum);
    if (is_first) {
      SCANN_LOG_NOOP(WARNING, 1000000) << StrFormat(
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

class GmmUtilsImplInterface : public VirtualDestructor {
 public:
  static unique_ptr<GmmUtilsImplInterface> Create(
      const DistanceMeasure& distance, const Dataset& dataset,
      ConstSpan<DatapointIndex> subset,
      thread::ThreadPool* parallelization_pool);

  virtual size_t size() const = 0;

  virtual size_t dimensionality() const = 0;

  virtual Status GetCentroid(Datapoint<double>* centroid) const = 0;

  virtual DatapointPtr<double> GetPoint(size_t idx,
                                        Datapoint<double>* storage) const = 0;

  using IterateDatasetCallback = std::function<void(
      size_t offset, const DenseDataset<double>& dataset_batch)>;
  virtual void IterateDataset(thread::ThreadPool* parallelization_pool,
                              const IterateDatasetCallback& callback) const = 0;

  void DistancesFromPoint(DatapointPtr<double> center,
                          MutableSpan<double> distances) const {
    this->IterateDataset(
        parallelization_pool_,
        [&](size_t offset,
            const DenseDataset<double>& dataset_batch) SCANN_INLINE_LAMBDA {
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

  Status CheckDataDegeneracy() {
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

  Status CheckAllFinite() const {
    Status finite_check_status = OkStatus();
    IterateDataset(
        nullptr, [&finite_check_status](
                     size_t offset, const DenseDataset<double>& dataset_batch) {
          if (!finite_check_status.ok()) return;
          for (size_t i : IndicesOf(dataset_batch)) {
            Status status = VerifyAllFinite(dataset_batch[i].values_slice());
            if (!status.ok()) {
              finite_check_status = AnnotateStatus(
                  status, StrFormat("(within-batch dp idx = %d)", offset + i));
              break;
            }
          }
        });
    return finite_check_status;
  }

 private:
  template <typename T>
  static unique_ptr<GmmUtilsImplInterface> CreateTyped(
      const DistanceMeasure& distance, const Dataset& dataset,
      ConstSpan<DatapointIndex> subset,
      thread::ThreadPool* parallelization_pool);

  const DistanceMeasure* distance_;
  thread::ThreadPool* parallelization_pool_;
};

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

  void IterateDataset(thread::ThreadPool* parallelization_pool,
                      const IterateDatasetCallback& callback) const final {
    constexpr size_t kBatchSize = 128;
    ParallelFor<1>(
        SeqWithStride<kBatchSize>(subset_.size()), parallelization_pool,
        [&](size_t subset_idx) SCANN_INLINE_LAMBDA {
          const size_t batch_size =
              std::min(kBatchSize, subset_.size() - subset_idx);

          DenseDataset<double> dataset_batch;
          dataset_batch.set_dimensionality(dataset_.dimensionality());
          dataset_batch.Reserve(batch_size);
          Datapoint<double> dp;
          for (size_t offset : Seq(batch_size)) {
            const DatapointIndex dp_index = subset_[subset_idx + offset];
            dataset_.GetDenseDatapoint(dp_index, &dp);
            TF_CHECK_OK(dataset_batch.Append(dp.ToPtr(), ""));
          }
          callback(subset_idx, dataset_batch);
        });
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

  void IterateDataset(thread::ThreadPool* parallelization_pool,
                      const IterateDatasetCallback& callback) const final {
    if (IsSame<T, double>()) {
      callback(0, reinterpret_cast<const DenseDataset<double>&>(dataset_));
      return;
    }

    constexpr size_t kBatchSize = 128;
    ParallelFor<1>(
        SeqWithStride<kBatchSize>(dataset_.size()), parallelization_pool,
        [&](size_t offset) SCANN_INLINE_LAMBDA {
          const size_t batch_size =
              std::min(kBatchSize, dataset_.size() - offset);
          DenseDataset<double> dataset_batch;
          dataset_batch.set_dimensionality(dataset_.dimensionality());
          dataset_batch.Reserve(batch_size);
          Datapoint<double> storage;
          for (size_t j : Seq(batch_size)) {
            auto dptr = MaybeConvertDatapoint(dataset_[offset + j], &storage);
            TF_CHECK_OK(dataset_batch.Append(dptr, ""));
          }
          callback(offset, dataset_batch);
        });
  }

 private:
  const DenseDataset<T>& dataset_;
};

template <typename T>
unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::CreateTyped(
    const DistanceMeasure& distance, const Dataset& dataset,
    ConstSpan<DatapointIndex> subset,
    thread::ThreadPool* parallelization_pool) {
  DCHECK(subset.empty());
  auto* dense_dataset = dynamic_cast<const DenseDataset<T>*>(&dataset);
  CHECK(dense_dataset);
  return make_unique<DenseDatasetWrapper<T>>(*dense_dataset);
}

unique_ptr<GmmUtilsImplInterface> GmmUtilsImplInterface::Create(
    const DistanceMeasure& distance, const Dataset& dataset,
    ConstSpan<DatapointIndex> subset,
    thread::ThreadPool* parallelization_pool) {
  unique_ptr<GmmUtilsImplInterface> impl;
  if (dataset.IsDense() && subset.empty()) {
    impl = SCANN_CALL_FUNCTION_BY_TAG(
        dataset.TypeTag(), GmmUtilsImplInterface::CreateTyped, distance,
        dataset, subset, parallelization_pool);
  } else {
    impl = make_unique<GenericDatasetWithSubset>(dataset, subset);
  }
  impl->distance_ = &distance;
  impl->parallelization_pool_ = parallelization_pool;
  return impl;
}

namespace {

vector<pair<DatapointIndex, double>> UnbalancedPartitionAssignment(
    GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
    const DenseDataset<double>& centers, thread::ThreadPool* pool) {
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

vector<pair<DatapointIndex, double>> MinCostMaxFlowPartitionAssignment(
    GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
    const DenseDataset<double>& centers, thread::ThreadPool* pool) {
  LOG(ERROR) << "Min-cost max-flow based assignment not supported.";
}

vector<pair<DatapointIndex, double>> GreedyBalancedPartitionAssignment(
    GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
    const DenseDataset<double>& centers, thread::ThreadPool* pool) {
  LOG(ERROR) << "Greedy partition balancing not supported.";
}

GmmUtils::PartitionAssignmentFn GetPartitionAssignmentFn(
    GmmUtils::Options::PartitionAssignmentType type) {
  switch (type) {
    case GmmUtils::Options::UNBALANCED:
      return &UnbalancedPartitionAssignment;
    case GmmUtils::Options::GREEDY_BALANCED:
      return &GreedyBalancedPartitionAssignment;
    case GmmUtils::Options::MIN_COST_MAX_FLOW:
      return &MinCostMaxFlowPartitionAssignment;
    default:
      LOG(FATAL) << "Invalid partition assignment type.";
  }
}

}  // namespace

Status GmmUtils::GenericKmeans(
    const Dataset& dataset, const int32_t num_clusters,
    DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  return KMeansImpl(false, dataset, {}, num_clusters,
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions);
}
Status GmmUtils::GenericKmeans(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const int32_t num_clusters, DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  return KMeansImpl(false, dataset, subset, num_clusters,
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions);
}

Status GmmUtils::SphericalKmeans(
    const Dataset& dataset, const int32_t num_clusters,
    DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  return KMeansImpl(true, dataset, {}, num_clusters,
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions);
}
Status GmmUtils::SphericalKmeans(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const int32_t num_clusters, DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  return KMeansImpl(true, dataset, subset, num_clusters,
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions);
}

Status GmmUtils::SphericalKmeans(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const DenseDataset<double>& initial_centers,
    DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  *final_centers = initial_centers.Copy();
  return KMeansImpl(true, dataset, subset, initial_centers.size(),
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions, true);
}
Status GmmUtils::GenericKmeans(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const DenseDataset<double>& initial_centers,
    DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions) {
  *final_centers = initial_centers.Copy();
  return KMeansImpl(false, dataset, subset, initial_centers.size(),
                    GetPartitionAssignmentFn(opts_.partition_assignment_type),
                    final_centers, final_partitions, true);
}

Status GmmUtils::InitializeCenters(const Dataset& dataset,
                                   ConstSpan<DatapointIndex> subset,
                                   int32_t num_clusters,
                                   DenseDataset<double>* initial_centers) {
  switch (opts_.center_initialization_type) {
    case Options::KMEANS_PLUS_PLUS:
      return KMeansPPInitializeCenters(dataset, subset, num_clusters,
                                       initial_centers);
    case Options::RANDOM_INITIALIZATION:
      return RandomInitializeCenters(dataset, subset, num_clusters,
                                     initial_centers);
    case Options::MEAN_DISTANCE_INITIALIZATION:
      return MeanDistanceInitializeCenters(dataset, subset, num_clusters,
                                           initial_centers);
  }
}

Status GmmUtils::MeanDistanceInitializeCenters(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    int32_t num_clusters, DenseDataset<double>* initial_centers) {
  SCANN_RET_CHECK(initial_centers);
  initial_centers->clear();
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  SCANN_RETURN_IF_ERROR(impl->CheckAllFinite())
      << "Non-finite values detected in the initial dataset in "
         "GmmUtils::InitializeCenters.";

  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  centers.set_dimensionality(dataset.dimensionality());
  centers.Reserve(num_clusters);

  Datapoint<double> storage;
  SCANN_RETURN_IF_ERROR(impl->GetCentroid(&storage));
  DatapointPtr<double> last_center = storage.ToPtr();

  vector<DatapointIndex> sample_ids;
  vector<double> distances(dataset_size, 0.0);

  impl->DistancesFromPoint(last_center, MakeMutableSpan(distances));
  SCANN_RETURN_IF_ERROR(VerifyAllFinite(last_center.values_slice()))
      << "(Center Number = " << centers.size() << ")";
  double min_dist = 0.0;
  double sum = 0.0;
  for (size_t j : Seq(distances.size())) {
    SCANN_RET_CHECK(!std::isnan(distances[j]))
        << "NaN distances found (j = " << j << ").";
    SCANN_RET_CHECK(std::isfinite(distances[j]))
        << "Infinite distances found (j = " << j << ").";
    min_dist = std::min(min_dist, distances[j]);
    sum += distances[j];
  }

  if (min_dist < 0.0) {
    VLOG(1) << "Biasing to get rid of negative distances. (min_dist = "
            << min_dist << ")";
    BiasDistances(-min_dist, MakeMutableSpan(distances));
    sum += -min_dist * distances.size();
  }

  while (centers.size() < num_clusters) {
    for (DatapointIndex idx : sample_ids) {
      sum -= distances[idx];
      distances[idx] = 0.0;
    }

    SCANN_RET_CHECK(!std::isnan(sum)) << "NaN distances sum found.";
    SCANN_RET_CHECK(std::isfinite(sum)) << "Infinite distances sum found.";
    DatapointIndex sample_id = GetSample(&random_, distances, sum, true);
    sample_ids.push_back(sample_id);
    last_center = impl->GetPoint(sample_id, &storage);
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(storage.values()));
    centers.AppendOrDie(last_center, "");
  }

  centers.set_normalization_tag(dataset.normalization());
  *initial_centers = std::move(centers);
  return OkStatus();
}

Status GmmUtils::KMeansPPInitializeCenters(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    int32_t num_clusters, DenseDataset<double>* initial_centers) {
  SCANN_RET_CHECK(initial_centers);
  initial_centers->clear();
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  SCANN_RETURN_IF_ERROR(impl->CheckAllFinite())
      << "Non-finite values detected in the initial dataset in "
         "GmmUtils::InitializeCenters.";

  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  centers.set_dimensionality(dataset.dimensionality());
  centers.Reserve(num_clusters);

  Datapoint<double> storage;
  SCANN_RETURN_IF_ERROR(impl->GetCentroid(&storage));
  DatapointPtr<double> last_center = storage.ToPtr();

  vector<DatapointIndex> sample_ids;
  vector<double> distances(dataset_size, 0.0);
  vector<double> temp(dataset_size);
  while (centers.size() < num_clusters) {
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(last_center.values_slice()))
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
    DatapointIndex sample_id =
        GetSample(&random_, distances, sum, centers.empty());
    sample_ids.push_back(sample_id);
    last_center = impl->GetPoint(sample_id, &storage);
    SCANN_RETURN_IF_ERROR(VerifyAllFinite(storage.values()));
    centers.AppendOrDie(last_center, "");
  }

  centers.set_normalization_tag(dataset.normalization());
  *initial_centers = std::move(centers);
  return OkStatus();
}

Status GmmUtils::RandomInitializeCenters(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    int32_t num_clusters, DenseDataset<double>* initial_centers) {
  SCANN_RET_CHECK(initial_centers);
  initial_centers->clear();
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
  SCANN_RETURN_IF_ERROR(impl->CheckAllFinite())
      << "Non-finite values detected in the initial dataset in "
         "GmmUtils::InitializeCenters.";

  const size_t dataset_size = impl->size();
  if (dataset_size < num_clusters) {
    return InvalidArgumentError(StrFormat(
        "Number of points (%d) is less than the number of clusters (%d).",
        dataset_size, num_clusters));
  }

  DenseDataset<double> centers;
  Datapoint<double> storage;
  centers.set_dimensionality(dataset.dimensionality());
  centers.Reserve(num_clusters);
  absl::flat_hash_set<DatapointIndex> center_ids;
  while (center_ids.size() < num_clusters) {
    center_ids.insert(absl::Uniform<DatapointIndex>(random_, 0, dataset_size));
  }
  for (DatapointIndex idx : center_ids) {
    SCANN_RETURN_IF_ERROR(centers.Append(impl->GetPoint(idx, &storage)));
  }

  centers.set_normalization_tag(dataset.normalization());
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

SCANN_OUTLINE Status GmmUtils::KMeansImpl(
    bool spherical, const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    int32_t num_clusters, PartitionAssignmentFn partition_assignment_fn,
    DenseDataset<double>* final_centers,
    vector<vector<DatapointIndex>>* final_partitions,
    bool preinitialized_centers) {
  DCHECK(final_centers);
  if (dataset.IsDense() && subset.size() == dataset.size() &&
      IsStdIota(subset)) {
    subset = ConstSpan<DatapointIndex>();
  }
  DenseDataset<double> centers;
  if (preinitialized_centers) {
    centers = std::move(*final_centers);
  } else {
    SCANN_RETURN_IF_ERROR(
        InitializeCenters(dataset, subset, num_clusters, &centers));
  }
  SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
      << "Initial centers contain NaN/infinity.";

  if (opts_.max_iterations <= 0) {
    return InvalidArgumentError(
        "Zero or negative iterations specified in GmmUtils::Options!");
  }
  if (opts_.min_cluster_size <= 0) {
    return InvalidArgumentError(
        "Zero or negative min cluster size specified in "
        "GmmUtils::Options!");
  }
  auto impl = GmmUtilsImplInterface::Create(*distance_, dataset, subset,
                                            opts_.parallelization_pool.get());
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

  if (spherical) {
    if (dataset.normalization() != UNITL2NORM) {
      return InvalidArgumentError("Input vectors must be unit L2-norm.");
    }
    if (centers.normalization() != UNITL2NORM) {
      return InvalidArgumentError("Initial centers must be unit L2-norm.");
    }
  }

  if (!impl->CheckDataDegeneracy().ok()) {
    SCANN_LOG_NOOP(WARNING, 1000000) << StrFormat(
        "All %d points are exactly the same in the partition.", dataset_size);
  }

  vector<double> old_means(num_clusters, -1.0);
  vector<double> new_means(num_clusters, -1.0);

  vector<uint32_t> partition_sizes(num_clusters);
  vector<pair<uint32_t, double>> top1_results;

  thread::ThreadPool* pool = opts_.parallelization_pool.get();
  for (size_t iteration : Seq(opts_.max_iterations + 1)) {
    top1_results =
        partition_assignment_fn(impl.get(), *distance_, centers, pool);
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
    if (iteration == opts_.max_iterations) {
      VLOG(1) << StrFormat("Exiting without converging after %d iterations.",
                           iteration);
      break;
    }

    if (opts_.parallel_cost_multiplier == 1.0) {
      SCANN_RETURN_IF_ERROR(RecomputeCentroidsSimple(
          top1_results, impl.get(), partition_sizes, spherical, &centers));
      SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
          << "RecomputeCentroidsSimple";
    } else {
      SCANN_RETURN_IF_ERROR(RecomputeCentroidsWithParallelCostMultiplier(
          top1_results, impl.get(), partition_sizes, spherical, &centers));
      SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
          << "RecomputeCentroidsWithParallelCostMultiplier";
    }

    switch (opts_.center_reassignment_type) {
      case Options::RANDOM_REASSIGNMENT:
        SCANN_RETURN_IF_ERROR(
            RandomReinitializeCenters(top1_results, impl.get(), partition_sizes,
                                      spherical, &centers, &new_means));
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
            << "RandomReinitializeCenters";
        break;
      case Options::SPLIT_LARGEST_CLUSTERS:
        SCANN_RETURN_IF_ERROR(SplitLargeClusterReinitialization(
            partition_sizes, spherical, &centers, &new_means));
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
            << "SplitLargeClusterReinitialization";
        break;
      case Options::PCA_SPLITTING:
        SCANN_RETURN_IF_ERROR(
            PCAKmeansReinitialization(top1_results, impl.get(), partition_sizes,
                                      spherical, &centers, &new_means));
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(centers.data()))
            << "PCAKmeansReinitialization";
        break;
    }
  }

  if (final_partitions) {
    vector<vector<DatapointIndex>> partitions(num_clusters);
    for (size_t c : Seq(num_clusters)) {
      partitions[c].reserve(partition_sizes[c]);
    }

    for (size_t j : Seq(dataset_size)) {
      const uint32_t cluster_idx = top1_results[j].first;
      const DatapointIndex dp_idx = subset.empty() ? j : subset[j];
      partitions[cluster_idx].push_back(dp_idx);
    }
    *final_partitions = std::move(partitions);
  }

  *final_centers = std::move(centers);
  if (spherical && opts_.parallel_cost_multiplier > 1.0) {
    SCANN_RETURN_IF_ERROR(final_centers->NormalizeUnitL2());
  }
  return OkStatus();
}

StatusOr<double> GmmUtils::ComputeSpillingThreshold(
    const Dataset& dataset, ConstSpan<DatapointIndex> subset,
    const DenseDataset<double>& centers,
    const DatabaseSpillingConfig::SpillingType spilling_type,
    const float total_spill_factor, const uint32_t max_centers) {
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

  const uint32_t max_neighbors = std::min(centers.size(), max_centers);

  vector<double> spills;

  Datapoint<double> storage;
  vector<double> distances(centers.size());
  for (size_t j : Seq(dataset_size)) {
    DenseDistanceOneToMany(*distance_, impl->GetPoint(j, &storage), centers,
                           MakeMutableSpan(distances),
                           opts_.parallelization_pool.get());
    gtl::TopN<double, std::less<double>> top_n(max_neighbors);
    top_n.reserve(max_neighbors + 1);
    for (double dist : distances) {
      top_n.push(dist);
    }
    auto top_items = *top_n.Extract();

    if (spilling_type == DatabaseSpillingConfig::ADDITIVE) {
      for (uint32_t k : Seq(1, max_neighbors)) {
        spills.push_back(top_items[k] - top_items[0]);
      }
    } else {
      DCHECK_EQ(spilling_type, DatabaseSpillingConfig::MULTIPLICATIVE);

      for (uint32_t k : Seq(1, max_neighbors)) {
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

Status GmmUtils::RecomputeCentroidsSimple(
    ConstSpan<pair<uint32_t, double>> top1_results, GmmUtilsImplInterface* impl,
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids) const {
  const size_t dataset_size = impl->size();
  const size_t dimensionality = impl->dimensionality();
  std::fill(centroids->mutable_data().begin(), centroids->mutable_data().end(),
            0.0);
  Datapoint<double> storage;
  for (size_t j : Seq(dataset_size)) {
    const uint32_t cluster_idx = top1_results[j].first;
    MutableSpan<double> mut_centroid = centroids->mutable_data(cluster_idx);
    ConstSpan<double> datapoint = impl->GetPoint(j, &storage).values_slice();
    SCANN_RET_CHECK_EQ(mut_centroid.size(), dimensionality);
    SCANN_RET_CHECK_EQ(datapoint.size(), dimensionality);
    for (size_t jj : Seq(dimensionality)) {
      mut_centroid[jj] += datapoint[jj];
    }
  }

  for (size_t c : IndicesOf(*centroids)) {
    MutableSpan<double> mut_centroid = centroids->mutable_data(c);
    const double divisor = spherical ? std::sqrt(SquaredL2Norm((*centroids)[c]))
                                     : static_cast<double>(partition_sizes[c]);
    if (divisor == 0) {
      SCANN_LOG_NOOP(WARNING, 1) << "Could not normalize centroid due to zero "
                                    "norm or empty partition.";
      continue;
    }
    const double multiplier = 1.0 / divisor;
    for (size_t jj : Seq(dimensionality)) {
      mut_centroid[jj] *= multiplier;
    }
  }
  return OkStatus();
}

Status GmmUtils::RandomReinitializeCenters(
    ConstSpan<pair<uint32_t, double>> top1_results, GmmUtilsImplInterface* impl,
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids, std::vector<double>* convergence_means) {
  Datapoint<double> storage;
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

    ConstSpan<double> rand_point =
        impl->GetPoint(rand_idx, &storage).values_slice();
    ConstSpan<double> old_center = (*centroids)[cluster_idx].values_slice();
    MutableSpan<double> mut_centroid = centroids->mutable_data(c);
    SCANN_RET_CHECK_EQ(rand_point.size(), dimensionality);
    SCANN_RET_CHECK_EQ(old_center.size(), dimensionality);
    SCANN_RET_CHECK_EQ(mut_centroid.size(), dimensionality);
    for (size_t jj : Seq(dimensionality)) {
      mut_centroid[jj] = old_center[jj] +
                         opts_.perturbation * (rand_point[jj] - old_center[jj]);
    }

    if (spherical) {
      const double norm = std::sqrt(SquaredL2Norm(centroids->at(c)));
      if (norm == 0) {
        SCANN_LOG_NOOP(WARNING, 1)
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

Status GmmUtils::SplitLargeClusterReinitialization(
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids, std::vector<double>* convergence_means) {
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
    MutableSpan<double> big_cluster =
        centroids->mutable_data(partition_permutation[big_cluster_idx]);
    MutableSpan<double> small_cluster =
        centroids->mutable_data(partition_permutation[small_cluster_idx]);
    for (const auto& d : Seq(dim)) {
      small_cluster[d] = big_cluster[d] + rand_direction[d];
      big_cluster[d] = big_cluster[d] - rand_direction[d];
    }
    if (spherical) {
      double big_cluster_norm =
          1.0f / sqrt(SquaredL2Norm(
                     centroids->at(partition_permutation[big_cluster_idx])));
      double small_cluster_norm =
          1.0f / sqrt(SquaredL2Norm(
                     centroids->at(partition_permutation[small_cluster_idx])));
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

Status GmmUtils::PCAKmeansReinitialization(
    ConstSpan<pair<uint32_t, double>> top1_results, GmmUtilsImplInterface* impl,
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids,
    std::vector<double>* convergence_means) const {
  using Eigen::aligned_allocator;
  using Eigen::JacobiSVD;
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::RowVectorXd;
  using Eigen::VectorXd;

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

  std::vector<MatrixXd, aligned_allocator<MatrixXd>> covariances(
      clusters_to_split.size());

  std::vector<VectorXd, aligned_allocator<VectorXd>> normalized_centroids(
      clusters_to_split.size());

  for (const auto& i : Seq(covariances.size())) {
    covariances[i].setZero(dim, dim);
    if (spherical) {
      Map<const RowVectorXd> centroid(
          centroids->at(partition_permutation[i]).values(), dim);
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

    Map<const RowVectorXd> centroid(centroids->at(cluster_idx).values(), dim);
    impl->GetPoint(i, &storage);
    Map<const RowVectorXd> dp(storage.values().data(), dim);
    VectorXd x = dp - centroid;
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
    const uint32_t num_split_directions =
        std::min(opts_.max_power_of_2_split,
                 32 - bits::CountLeadingZeros32(multiple_of_avg));

    covariances[i] /= sorted_partition_sizes[i];

    JacobiSVD<MatrixXd> svd(covariances[i], Eigen::ComputeThinU);
    std::vector<VectorXd, aligned_allocator<VectorXd>> split_directions;
    for (int j = 0; j < dim && split_directions.size() < num_split_directions;
         ++j) {
      const double stdev = std::sqrt(svd.singularValues()[j]);
      const double scaling_factor =
          std::max(stdev * opts_.perturbation, DBL_EPSILON * dim);
      split_directions.push_back(svd.matrixU().col(j) * scaling_factor);
    }

    const uint64_t combinatoric_limit = 1ULL << split_directions.size();
    VectorXd old_centroid(dim);
    VectorXd centroid_storage(dim);
    auto old_centroid_span = centroids->mutable_data(partition_permutation[i]);
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
      auto override_centroid = centroids->mutable_data(small_cluster_index);

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
    SCANN_RETURN_IF_ERROR(centroids->NormalizeUnitL2());
  }
  absl::Time cov_end = absl::Now();
  if (!clusters_to_split.empty()) {
    LOG(INFO) << "Spent " << absl::ToDoubleSeconds(cov_end - cov_start)
              << " seconds for computing covariances";
  }

  return OkStatus();
}

Status GmmUtils::RecomputeCentroidsWithParallelCostMultiplier(
    ConstSpan<pair<uint32_t, double>> top1_results, GmmUtilsImplInterface* impl,
    ConstSpan<uint32_t> partition_sizes, bool spherical,
    DenseDataset<double>* centroids) const {
  const double parallel_cost_multiplier = opts_.parallel_cost_multiplier;
  SCANN_RET_CHECK_NE(parallel_cost_multiplier, 1.0);
  const size_t dimensionality = impl->dimensionality();

  vector<double> mean_vec(centroids->data().size());
  DenseDataset<double> means(std::move(mean_vec), centroids->size());
  SCANN_RETURN_IF_ERROR(RecomputeCentroidsSimple(
      top1_results, impl, partition_sizes, false, &means));

  vector<std::vector<DatapointIndex>> assignments(centroids->size());
  for (DatapointIndex dp_idx : IndicesOf(top1_results)) {
    const size_t partition_idx = top1_results[dp_idx].first;
    assignments[partition_idx].push_back(dp_idx);
  }

  Datapoint<double> storage;
  Eigen::MatrixXd outer_prodsums;
  auto add_outer_product = [&outer_prodsums](ConstSpan<double> vec) {
    DCHECK_EQ(vec.size(), outer_prodsums.cols());
    DCHECK_EQ(vec.size(), outer_prodsums.rows());
    Eigen::Map<const Eigen::VectorXd> vm(vec.data(), vec.size());
    const double denom = vm.transpose() * vm;
    if (denom > 0) {
      outer_prodsums += (vm * vm.transpose()) / denom;
    }
  };
  const double lambda = 1.0 / parallel_cost_multiplier;
  for (size_t partition_idx : IndicesOf(assignments)) {
    MutableSpan<double> mut_centroid = centroids->mutable_data(partition_idx);
    if (assignments[partition_idx].empty()) {
      std::fill(mut_centroid.begin(), mut_centroid.end(), 0.0);
      continue;
    }
    outer_prodsums = Eigen::MatrixXd::Zero(dimensionality, dimensionality);
    for (DatapointIndex dp_idx : assignments[partition_idx]) {
      ConstSpan<double> datapoint =
          impl->GetPoint(dp_idx, &storage).values_slice();
      add_outer_product(datapoint);
    }
    outer_prodsums *= (1.0 - lambda) / assignments[partition_idx].size();

    for (size_t i : Seq(dimensionality)) {
      outer_prodsums(i, i) += lambda;
    }

    Eigen::Map<const Eigen::VectorXd> mean(means[partition_idx].values(),
                                           dimensionality);
    Eigen::VectorXd centroid = outer_prodsums.inverse() * mean;
    for (size_t i : Seq(dimensionality)) {
      mut_centroid[i] = centroid(i);
    }
  }
  return OkStatus();
}

}  // namespace scann_ops
}  // namespace tensorflow
