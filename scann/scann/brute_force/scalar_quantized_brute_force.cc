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

#include "scann/brute_force/scalar_quantized_brute_force.h"

#include "scann/base/restrict_allowlist.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"

#include "absl/memory/memory.h"
#include "scann/oss_wrappers/scann_status_builder.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace scann_ops {

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
  quantized_dataset_ = std::move(quantization_results.quantized_dataset);
  inverse_multiplier_by_dimension_ =
      std::move(quantization_results.inverse_multiplier_by_dimension);
  const auto distance_tag = distance->specially_optimized_distance_tag();
  auto distance_tag_status = CheckValidDistanceTag(distance_tag);
  if (!distance_tag_status.ok()) {
    LOG(FATAL) << distance_tag_status;
  }
  if (distance_tag == DistanceMeasure::SQUARED_L2) {
    squared_l2_norms_.resize(dataset->size());
    for (DatapointIndex i = 0; i < dataset->size(); ++i) {
      squared_l2_norms_[i] = SquaredL2Norm((*dataset)[i]);
    }
  }
}

ScalarQuantizedBruteForceSearcher::ScalarQuantizedBruteForceSearcher(
    shared_ptr<const DistanceMeasure> distance, vector<float> squared_l2_norms,
    DenseDataset<int8_t> quantized_dataset,
    vector<float> inverse_multiplier_by_dimension,
    int32_t default_num_neighbors, float default_epsilon)
    : SingleMachineSearcherBase<float>(nullptr, default_num_neighbors,
                                       default_epsilon),
      distance_(distance),
      squared_l2_norms_(std::move(squared_l2_norms)),
      quantized_dataset_(std::move(quantized_dataset)),
      inverse_multiplier_by_dimension_(
          std::move(inverse_multiplier_by_dimension)) {
  TF_CHECK_OK(this->set_docids(quantized_dataset_.ReleaseDocids()));
}

StatusOr<vector<float>>
ScalarQuantizedBruteForceSearcher::ComputeSquaredL2NormsFromQuantizedDataset(
    const DenseDataset<int8_t>& quantized,
    const vector<float>& inverse_multipliers) {
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
        DenseDataset<int8_t> quantized, vector<float> inverse_multipliers,
        vector<float> squared_l2_norms, int32_t default_num_neighbors,
        float default_epsilon) {
  const auto distance_tag = distance->specially_optimized_distance_tag();
  SCANN_RETURN_IF_ERROR(CheckValidDistanceTag(distance_tag));
  if (distance_tag == DistanceMeasure::SQUARED_L2 && !quantized.empty() &&
      squared_l2_norms.empty()) {
    SCANN_LOG_NOOP(INFO, 1)
        << "squared_l2_norms are not loaded, and they will be computed.";
    TF_ASSIGN_OR_RETURN(squared_l2_norms,
                        ComputeSquaredL2NormsFromQuantizedDataset(
                            quantized, inverse_multipliers));
  }

  return absl::make_unique<ScalarQuantizedBruteForceSearcher>(
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
    TF_ASSIGN_OR_RETURN(
        squared_l2_norms,
        ComputeSquaredL2NormsFromQuantizedDataset(
            quantization_results.quantized_dataset,
            quantization_results.inverse_multiplier_by_dimension));
  }

  return absl::make_unique<ScalarQuantizedBruteForceSearcher>(
      distance, std::move(squared_l2_norms),
      std::move(quantization_results.quantized_dataset),
      std::move(quantization_results.inverse_multiplier_by_dimension),
      default_num_neighbors, default_epsilon);
}

ScalarQuantizedBruteForceSearcher::~ScalarQuantizedBruteForceSearcher() {}

Status ScalarQuantizedBruteForceSearcher::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (datapoint_index_to_crowding_attribute.size() !=
      quantized_dataset_.size()) {
    return InvalidArgumentError(absl::StrCat(
        "datapoint_index_to_crowding_attribute must have size equal to "
        "number of datapoints.  (",
        datapoint_index_to_crowding_attribute.size(), " vs. ",
        quantized_dataset_.size(), "."));
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
    if (inverse_multiplier_by_dimension_.empty())
      return InvalidArgumentError(
          "TreeScalarQuantizationPreprocessedQuery is not specified and "
          "inverse "
          "multipliers are empty.");
    preproc_buf = PrepareForAsymmetricScalarQuantizedDotProduct(
        query, inverse_multiplier_by_dimension_);
    preprocessed = MakeDatapointPtr(preproc_buf.get(), query.nonzero_entries());
  }

  if (params.restricts_enabled()) {
    return UnimplementedError("Restricts not supported.");
  } else {
    auto dot_products_ptr =
        static_cast<float*>(malloc(quantized_dataset_.size() * sizeof(float)));
    MutableSpan<float> dot_products(dot_products_ptr,
                                    quantized_dataset_.size());
    DenseDotProductDistanceOneToManyInt8Float(preprocessed, quantized_dataset_,
                                              dot_products);
    Status status =
        PostprocessDistances<float>(query, params, dot_products, result);
    free(dot_products_ptr);
    return status;
  }
}

template <typename ResultElem>
Status ScalarQuantizedBruteForceSearcher::PostprocessDistances(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<ResultElem> dot_products, NNResultsVector* result) const {
  switch (distance_->specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
      return PostprocessDistancesImpl(
          query, params, dot_products,
          [](float dot_product, DatapointIndex i) { return dot_product; },
          result);
    case DistanceMeasure::COSINE:

      return PostprocessDistancesImpl(
          query, params, dot_products,
          [](float dot_product, DatapointIndex i) {
            return 1.0f + dot_product;
          },
          result);
    case DistanceMeasure::SQUARED_L2: {
      ConstSpan<float> squared_norms = squared_l2_norms_;
      const float query_squared_l2_norm = SquaredL2Norm(query);
      return PostprocessDistancesImpl(
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

template <typename DistanceFunctor, typename ResultElem>
Status ScalarQuantizedBruteForceSearcher::PostprocessDistancesImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<ResultElem> dot_products, DistanceFunctor distance_functor,
    NNResultsVector* result) const {
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else {
    TopNeighbors<float> top_n(params.pre_reordering_num_neighbors());
    SCANN_RETURN_IF_ERROR(PostprocessTopNImpl(query, params, dot_products,
                                              distance_functor, &top_n));
    *result = top_n.ExtractUnsorted();
  }
  return OkStatus();
}

template <typename DistanceFunctor, typename TopN>
Status ScalarQuantizedBruteForceSearcher::PostprocessTopNImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<float> dot_products, DistanceFunctor distance_functor,
    TopN* top_n_ptr) const {
  DCHECK(!params.restricts_enabled());
  TopN top_n = std::move(*top_n_ptr);
  const float epsilon = params.pre_reordering_epsilon();
  float min_keep_distance = epsilon;
  for (DatapointIndex i = 0; i < dot_products.size(); ++i) {
    const float distance = distance_functor(dot_products[i], i);
    if (distance <= min_keep_distance) {
      top_n.push(std::make_pair(i, distance));
      if (top_n.full()) {
        min_keep_distance = top_n.approx_bottom().second;
      }
    }
  }
  *top_n_ptr = std::move(top_n);
  return OkStatus();
}

template <typename DistanceFunctor, typename TopN>
Status ScalarQuantizedBruteForceSearcher::PostprocessTopNImpl(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<pair<DatapointIndex, float>> dot_products,
    DistanceFunctor distance_functor, TopN* top_n_ptr) const {
  DCHECK(params.restricts_enabled());
  TopN top_n = std::move(*top_n_ptr);
  const float epsilon = params.pre_reordering_epsilon();
  float min_keep_distance = epsilon;
  for (const auto& pair : dot_products) {
    const DatapointIndex dp_idx = pair.first;
    const float distance = distance_functor(pair.second, dp_idx);
    if (distance <= min_keep_distance) {
      top_n.push(std::make_pair(dp_idx, distance));
      if (top_n.full()) {
        min_keep_distance = top_n.approx_bottom().second;
      }
    }
  }
  *top_n_ptr = std::move(top_n);
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

StatusOr<SingleMachineFactoryOptions>
ScalarQuantizedBruteForceSearcher::ExtractSingleMachineFactoryOptions() {
  TF_ASSIGN_OR_RETURN(
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
          quantized_dataset_, inverse_multiplier_by_dimension_,
          squared_l2_norms_, true));
  return opts;
}

}  // namespace scann_ops
}  // namespace tensorflow
