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

#include "scann/brute_force/scalar_quantized_brute_force.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {

Status CheckValidDistanceTag(
    AbsDotProductDistance::SpeciallyOptimizedDistanceTag distance_tag) {
  if (distance_tag != DistanceMeasure::DOT_PRODUCT &&
      distance_tag != DistanceMeasure::COSINE &&
      distance_tag != DistanceMeasure::SQUARED_L2) {
    return InvalidArgumentError(
        "Distance measure must be DotProductDistance, "
        "CosineDistance or SquaredL2Distance for "
        "ScalarQuantizedBruteForceSearcher.");
  }
  return OkStatus();
}

ScalarQuantizedBruteForceSearcher::ScalarQuantizedBruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const DenseDataset<float>> dataset,
    const int32_t default_pre_reordering_num_neighbors,
    const float default_pre_reordering_epsilon, Options opts)
    : SingleMachineSearcherBase<float>(dataset,
                                       default_pre_reordering_num_neighbors,
                                       default_pre_reordering_epsilon),
      distance_(distance),
      opts_(opts) {
  ScalarQuantizationResults quantization_results = ScalarQuantizeFloatDataset(
      *dataset, opts.multiplier_quantile, opts.noise_shaping_threshold);
  quantized_dataset_ = make_shared<DenseDataset<int8_t>>(
      std::move(quantization_results.quantized_dataset));
  inverse_multiplier_by_dimension_ = make_shared<vector<float>>(
      std::move(quantization_results.inverse_multiplier_by_dimension));
  const auto distance_tag = distance->specially_optimized_distance_tag();
  auto distance_tag_status = CheckValidDistanceTag(distance_tag);
  if (!distance_tag_status.ok()) {
    LOG(FATAL) << distance_tag_status;
  }

  if (distance_tag == DistanceMeasure::SQUARED_L2) {
    vector<float> squared_l2_norms(dataset->size());
    for (DatapointIndex i = 0; i < dataset->size(); ++i) {
      squared_l2_norms[i] = SquaredL2Norm((*dataset)[i]);
    }
    squared_l2_norms_ = make_shared<vector<float>>(std::move(squared_l2_norms));
  }
}

ScalarQuantizedBruteForceSearcher::ScalarQuantizedBruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<vector<float>> squared_l2_norms,
    shared_ptr<const DenseDataset<int8_t>> quantized_dataset,
    shared_ptr<const vector<float>> inverse_multiplier_by_dimension,
    int32_t default_num_neighbors, float default_epsilon)
    : SingleMachineSearcherBase<float>(nullptr, default_num_neighbors,
                                       default_epsilon),
      distance_(distance),
      squared_l2_norms_(std::move(squared_l2_norms)),
      quantized_dataset_(std::move(quantized_dataset)),
      inverse_multiplier_by_dimension_(
          std::move(inverse_multiplier_by_dimension)) {
  QCHECK_OK(this->set_docids(quantized_dataset_->docids()));
}

StatusOr<vector<float>>
ScalarQuantizedBruteForceSearcher::ComputeSquaredL2NormsFromQuantizedDataset(
    const DenseDataset<int8_t>& quantized,
    absl::Span<const float> inverse_multipliers) {
  if (quantized.dimensionality() != inverse_multipliers.size())
    return InvalidArgumentError(absl::StrCat(
        "The dimension of quantized dataset ", quantized.dimensionality(),
        " is not equal to the size of inverse multiplier vector ",
        inverse_multipliers.size()));

  vector<float> squared_l2_norms(quantized.size(), 0.0);
  for (DatapointIndex i = 0; i < quantized.size(); ++i) {
    const auto& dp = quantized[i];
    const auto* values = dp.values();
    for (DimensionIndex j = 0; j < dp.dimensionality(); ++j) {
      const auto dequantized = values[j] * inverse_multipliers[j];
      squared_l2_norms[i] += dequantized * dequantized;
    }
  }
  return squared_l2_norms;
}

StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
ScalarQuantizedBruteForceSearcher::
    CreateFromQuantizedDatasetAndInverseMultipliers(
        shared_ptr<const DistanceMeasure> distance,
        shared_ptr<const DenseDataset<int8_t>> quantized,
        shared_ptr<const vector<float>> inverse_multipliers,
        shared_ptr<vector<float>> squared_l2_norms,
        int32_t default_num_neighbors, float default_epsilon) {
  const auto distance_tag = distance->specially_optimized_distance_tag();
  SCANN_RETURN_IF_ERROR(CheckValidDistanceTag(distance_tag));
  if (distance_tag == DistanceMeasure::SQUARED_L2 && !quantized->empty() &&
      (!squared_l2_norms || squared_l2_norms->empty())) {
    LOG_FIRST_N(INFO, 1)
        << "squared_l2_norms are not loaded, and they will be computed.";
    SCANN_ASSIGN_OR_RETURN(auto squared_l2_norms_vec,
                           ComputeSquaredL2NormsFromQuantizedDataset(
                               *quantized, *inverse_multipliers));
    squared_l2_norms =
        make_shared<vector<float>>(std::move(squared_l2_norms_vec));
  }

  return std::make_unique<ScalarQuantizedBruteForceSearcher>(
      distance, std::move(squared_l2_norms), std::move(quantized),
      std::move(inverse_multipliers), default_num_neighbors, default_epsilon);
}

StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
ScalarQuantizedBruteForceSearcher::CreateWithFixedRange(
    shared_ptr<const DistanceMeasure> distance,
    shared_ptr<const DenseDataset<float>> dataset,
    ConstSpan<float> abs_thresholds_for_each_dimension,
    int32_t default_num_neighbors, float default_epsilon) {
  const auto distance_tag = distance->specially_optimized_distance_tag();
  SCANN_RETURN_IF_ERROR(CheckValidDistanceTag(distance_tag));

  DCHECK_EQ(dataset->dimensionality(),
            abs_thresholds_for_each_dimension.size());
  std::vector<float> multipliers(dataset->dimensionality());

  for (auto i : Seq(multipliers.size())) {
    multipliers[i] = abs_thresholds_for_each_dimension[i] == 0.0f
                         ? 1.0f
                         : numeric_limits<int8_t>::max() /
                               abs_thresholds_for_each_dimension[i];
  }
  auto quantization_results = ScalarQuantizeFloatDatasetWithMultipliers(
      *dataset, std::move(multipliers));

  vector<float> squared_l2_norms;
  if (distance_tag == DistanceMeasure::SQUARED_L2 && !dataset->empty()) {
    SCANN_ASSIGN_OR_RETURN(
        squared_l2_norms,
        ComputeSquaredL2NormsFromQuantizedDataset(
            quantization_results.quantized_dataset,
            quantization_results.inverse_multiplier_by_dimension));
  }

  return std::make_unique<ScalarQuantizedBruteForceSearcher>(
      distance, make_shared<vector<float>>(std::move(squared_l2_norms)),
      make_shared<DenseDataset<int8_t>>(
          std::move(quantization_results.quantized_dataset)),
      make_shared<vector<float>>(
          std::move(quantization_results.inverse_multiplier_by_dimension)),
      default_num_neighbors, default_epsilon);
}

StatusOr<const SingleMachineSearcherBase<float>*>
ScalarQuantizedBruteForceSearcher::CreateBruteForceSearcher(
    const DistanceMeasureConfig& distance_config,
    unique_ptr<SingleMachineSearcherBase<float>>* storage) const {
  auto base_result = SingleMachineSearcherBase<float>::CreateBruteForceSearcher(
      distance_config, storage);
  if (base_result.ok()) {
    return base_result;
  }
  return this;
}

ScalarQuantizedBruteForceSearcher::~ScalarQuantizedBruteForceSearcher() {}

Status ScalarQuantizedBruteForceSearcher::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (datapoint_index_to_crowding_attribute.size() !=
      quantized_dataset_->size()) {
    return InvalidArgumentError(absl::StrCat(
        "datapoint_index_to_crowding_attribute must have size equal to "
        "number of datapoints.  (",
        datapoint_index_to_crowding_attribute.size(), " vs. ",
        quantized_dataset_->size(), "."));
  }
  return OkStatus();
}

Status ScalarQuantizedBruteForceSearcher::FindNeighborsImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    NNResultsVector* result) const {
  DCHECK(result);
  if (!query.IsDense()) {
    return InvalidArgumentError(
        "ScalarQuantizedBruteForceSearcher only works with dense data.");
  }
  if (query.dimensionality() != quantized_dataset_->dimensionality()) {
    return FailedPreconditionError(absl::StrFormat(
        "Query dimensionality (%d) does not match quantized database "
        "dimensionality (%d)",
        query.dimensionality(), quantized_dataset_->dimensionality()));
  }
  DatapointPtr<float> preprocessed;
  unique_ptr<float[]> preproc_buf;
  const auto* tree_sq_preproc_query =
      params.searcher_specific_optional_parameters();
  if (tree_sq_preproc_query) {
    const auto* casted =
        down_cast<const TreeScalarQuantizationPreprocessedQuery*>(
            tree_sq_preproc_query);
    DCHECK(casted)
        << "Downcast to TreeScalarQuantizationPreprocessedQuery failed.";
    preprocessed =
        MakeDatapointPtr(casted->PreprocessedQuery(), query.nonzero_entries());
  } else {
    if (inverse_multiplier_by_dimension_->empty())
      return InvalidArgumentError(
          "TreeScalarQuantizationPreprocessedQuery is not specified and "
          "inverse "
          "multipliers are empty.");
    preproc_buf = PrepareForAsymmetricScalarQuantizedDotProduct(
        query, *inverse_multiplier_by_dimension_);
    preprocessed = MakeDatapointPtr(preproc_buf.get(), query.nonzero_entries());
  }

  const bool use_min_distance =
      min_distance_ > -numeric_limits<float>::infinity();
  if (params.restricts_enabled()) {
    return UnimplementedError("Restricts not supported.");
  } else {
    auto dot_products_ptr =
        static_cast<float*>(malloc(quantized_dataset_->size() * sizeof(float)));
    MutableSpan<float> dot_products(dot_products_ptr,
                                    quantized_dataset_->size());
    DenseDotProductDistanceOneToManyInt8Float(preprocessed, *quantized_dataset_,
                                              dot_products);
    Status status = use_min_distance ? PostprocessDistances<true, float>(
                                           query, params, dot_products, result)
                                     : PostprocessDistances<false, float>(
                                           query, params, dot_products, result);
    free(dot_products_ptr);
    return status;
  }
}

template <bool kUseMinDistance, typename ResultElem>
Status ScalarQuantizedBruteForceSearcher::PostprocessDistances(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<ResultElem> dot_products, NNResultsVector* result) const {
  switch (distance_->specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
      return PostprocessDistancesImpl<kUseMinDistance>(
          query, params, dot_products,
          [](float dot_product, DatapointIndex i) { return dot_product; },
          result);
    case DistanceMeasure::COSINE:

      return PostprocessDistancesImpl<kUseMinDistance>(
          query, params, dot_products,
          [](float dot_product, DatapointIndex i) {
            return 1.0f + dot_product;
          },
          result);
    case DistanceMeasure::SQUARED_L2: {
      ConstSpan<float> squared_norms = *squared_l2_norms_;
      const float query_squared_l2_norm = SquaredL2Norm(query);
      return PostprocessDistancesImpl<kUseMinDistance>(
          query, params, dot_products,
          [&squared_norms, query_squared_l2_norm](float dot_product,
                                                  DatapointIndex i) {
            return query_squared_l2_norm + squared_norms[i] +
                   2.0f * dot_product;
          },
          result);
    }
    default:
      return FailedPreconditionError(
          "ScalarQuantizedBruteForceSearcher only works with "
          "SquaredL2Distance, CosineDistance and DotProductDistance.");
  }
  return OkStatus();
}

template <bool kUseMinDistance, typename DistanceFunctor, typename ResultElem>
Status ScalarQuantizedBruteForceSearcher::PostprocessDistancesImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<ResultElem> dot_products, DistanceFunctor distance_functor,
    NNResultsVector* result) const {
  const bool use_min_distance =
      min_distance_ > -numeric_limits<float>::infinity();
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    FastTopNeighbors<float> top_n(params.pre_reordering_num_neighbors(),
                                  params.pre_reordering_epsilon());
    if (use_min_distance) {
      SCANN_RETURN_IF_ERROR(PostprocessTopNImpl<true>(
          query, params, dot_products, distance_functor, &top_n));
    } else {
      SCANN_RETURN_IF_ERROR(PostprocessTopNImpl<false>(
          query, params, dot_products, distance_functor, &top_n));
    }
    top_n.FinishUnsorted(result);
  }
  return OkStatus();
}

template <bool kUseMinDistance, typename DistanceFunctor, typename TopN>
Status ScalarQuantizedBruteForceSearcher::PostprocessTopNImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<float> dot_products, DistanceFunctor distance_functor,
    TopN* top_n_ptr) const {
  DCHECK(!params.restricts_enabled());
  typename TopN::Mutator mutator;
  top_n_ptr->AcquireMutator(&mutator);
  float min_keep_distance = mutator.epsilon();
  const float min_distance = min_distance_;
  for (DatapointIndex i = 0; i < dot_products.size(); ++i) {
    const float distance = distance_functor(dot_products[i], i);
    if (distance <= min_keep_distance &&
        (!kUseMinDistance || distance >= min_distance)) {
      if (mutator.Push(i, distance)) {
        mutator.GarbageCollect();
        min_keep_distance = mutator.epsilon();
      }
    }
  }
  return OkStatus();
}

template <bool kUseMinDistance, typename DistanceFunctor, typename TopN>
Status ScalarQuantizedBruteForceSearcher::PostprocessTopNImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<pair<DatapointIndex, float>> dot_products,
    DistanceFunctor distance_functor, TopN* top_n_ptr) const {
  DCHECK(params.restricts_enabled());
  typename TopN::Mutator mutator;
  top_n_ptr->AcquireMutator(&mutator);
  float min_keep_distance = mutator.epsilon();
  const float min_distance = min_distance_;
  for (const auto& pair : dot_products) {
    const DatapointIndex dp_idx = pair.first;
    const float distance = distance_functor(pair.second, dp_idx);
    if (distance <= min_keep_distance &&
        (!kUseMinDistance || distance >= min_distance)) {
      if (mutator.Push(dp_idx, distance)) {
        mutator.GarbageCollect();
        min_keep_distance = mutator.epsilon();
      }
    }
  }
  return OkStatus();
}

StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
TreeScalarQuantizationPreprocessedQueryCreator::
    CreateLeafSearcherOptionalParameters(
        const DatapointPtr<float>& query) const {
  auto preprocessed_query = PrepareForAsymmetricScalarQuantizedDotProduct(
      query, inverse_multipliers_);
  return unique_ptr<SearcherSpecificOptionalParameters>(
      new TreeScalarQuantizationPreprocessedQuery(
          std::move(preprocessed_query)));
}

ConstSpan<float>
TreeScalarQuantizationPreprocessedQueryCreator::inverse_multipliers() const {
  return MakeConstSpan(inverse_multipliers_.data(),
                       inverse_multipliers_.size());
}

StatusOr<SingleMachineSearcherBase<float>::Mutator*>
ScalarQuantizedBruteForceSearcher::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<ScalarQuantizedBruteForceSearcher*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_,
        ScalarQuantizedBruteForceSearcher::Mutator::Create(mutable_this));
    SCANN_RETURN_IF_ERROR(mutator_->PrepareForBaseMutation(mutable_this));
  }
  return static_cast<SingleMachineSearcherBase::Mutator*>(mutator_.get());
}

StatusOr<SingleMachineFactoryOptions>
ScalarQuantizedBruteForceSearcher::ExtractSingleMachineFactoryOptions() {
  SCANN_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<float>::ExtractSingleMachineFactoryOptions());
  if (opts.pre_quantized_fixed_point != nullptr) {
    return InvalidArgumentError(
        "pre_quantized_fixed_point already exists. Either disable reordering "
        "or use float32 reordering, because scalar-quantized reordering with "
        "scalar-quantized brute force provides no benefit.");
  }
  opts.pre_quantized_fixed_point =
      make_shared<PreQuantizedFixedPoint>(CreatePreQuantizedFixedPoint(
          *quantized_dataset_, *inverse_multiplier_by_dimension_,
          *squared_l2_norms_, true));
  return opts;
}

StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
ScalarQuantizedBruteForceSearcher::
    CreateFromQuantizedDatasetAndInverseMultipliers(
        shared_ptr<const DistanceMeasure> distance,
        DenseDataset<int8_t> quantized, vector<float> inverse_multipliers,
        vector<float> squared_l2_norms, int32_t default_num_neighbors,
        float default_epsilon) {
  using std::make_shared;
  using std::move;
  return CreateFromQuantizedDatasetAndInverseMultipliers(
      move(distance), make_shared<DenseDataset<int8_t>>(move(quantized)),
      make_shared<vector<float>>(move(inverse_multipliers)),
      make_shared<vector<float>>(move(squared_l2_norms)), default_num_neighbors,
      default_epsilon);
}

}  // namespace research_scann
