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



#ifndef SCANN__BRUTE_FORCE_SCALAR_QUANTIZED_BRUTE_FORCE_H_
#define SCANN__BRUTE_FORCE_SCALAR_QUANTIZED_BRUTE_FORCE_H_

#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace scann_ops {

class ScalarQuantizedBruteForceSearcher final
    : public SingleMachineSearcherBase<float> {
 public:
  struct Options {
    float multiplier_quantile = 1.0f;
    float noise_shaping_threshold = NAN;
  };

  ScalarQuantizedBruteForceSearcher(
      shared_ptr<const DistanceMeasure> distance,
      shared_ptr<const DenseDataset<float>> dataset,
      const int32_t default_pre_reordering_num_neighbors,
      const float default_pre_reordering_epsilon, Options opts);

  ScalarQuantizedBruteForceSearcher(
      shared_ptr<const DistanceMeasure> distance,
      shared_ptr<const DenseDataset<float>> dataset,
      const int32_t default_pre_reordering_num_neighbors,
      const float default_pre_reordering_epsilon)
      : ScalarQuantizedBruteForceSearcher(
            std::move(distance), std::move(dataset),
            default_pre_reordering_num_neighbors,
            default_pre_reordering_epsilon, Options()) {}

  ~ScalarQuantizedBruteForceSearcher() override;

  bool supports_crowding() const final { return true; }

  ScalarQuantizedBruteForceSearcher(
      shared_ptr<const DistanceMeasure> distance,
      vector<float> squared_l2_norms, DenseDataset<int8_t> quantized_dataset,
      vector<float> inverse_multiplier_by_dimension,
      int32_t default_num_neighbors, float default_epsilon);

  static StatusOr<vector<float>> ComputeSquaredL2NormsFromQuantizedDataset(
      const DenseDataset<int8_t>& quantized,
      const vector<float>& inverse_multipliers);

  static StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
  CreateFromQuantizedDatasetAndInverseMultipliers(
      shared_ptr<const DistanceMeasure> distance,
      DenseDataset<int8_t> quantized, vector<float> inverse_multipliers,
      vector<float> squared_l2_norms, int32_t default_num_neighbors,
      float default_epsilon);

  static StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
  CreateWithFixedRange(shared_ptr<const DistanceMeasure> distance,
                       shared_ptr<const DenseDataset<float>> dataset,
                       ConstSpan<float> abs_thresholds_for_each_dimension,
                       int32_t default_num_neighbors, float default_epsilon);

  StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
      override;

 protected:
  Status FindNeighborsImpl(const DatapointPtr<float>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute) final;

 private:
  template <typename ResultElem>
  Status PostprocessDistances(const DatapointPtr<float>& query,
                              const SearchParameters& params,
                              ConstSpan<ResultElem> dot_products,
                              NNResultsVector* result) const;

  template <typename DistanceFunctor, typename ResultElem>
  Status PostprocessDistancesImpl(const DatapointPtr<float>& query,
                                  const SearchParameters& params,
                                  ConstSpan<ResultElem> dot_products,
                                  DistanceFunctor distance_functor,
                                  NNResultsVector* result) const;

  template <typename DistanceFunctor, typename TopN>
  Status PostprocessTopNImpl(const DatapointPtr<float>& query,
                             const SearchParameters& params,
                             ConstSpan<float> dot_products,
                             DistanceFunctor distance_functor,
                             TopN* top_n_ptr) const;

  template <typename DistanceFunctor, typename TopN>
  Status PostprocessTopNImpl(
      const DatapointPtr<float>& query, const SearchParameters& params,
      ConstSpan<pair<DatapointIndex, float>> dot_products,
      DistanceFunctor distance_functor, TopN* top_n_ptr) const;

  bool impl_needs_dataset() const override { return false; }

  shared_ptr<const DistanceMeasure> distance_;

  vector<float> squared_l2_norms_;

  DenseDataset<int8_t> quantized_dataset_;

  Options opts_;

  vector<float> inverse_multiplier_by_dimension_;

  mutable unique_ptr<Mutator> mutator_ = nullptr;
};

class TreeScalarQuantizationPreprocessedQuery final
    : public SearcherSpecificOptionalParameters {
 public:
  explicit TreeScalarQuantizationPreprocessedQuery(
      unique_ptr<float[]> preprocessed_query)
      : preprocessed_query_(std::move(preprocessed_query)) {}

  const float* PreprocessedQuery() const { return preprocessed_query_.get(); }

 private:
  const unique_ptr<float[]> preprocessed_query_;
};

class TreeScalarQuantizationPreprocessedQueryCreator final
    : public LeafSearcherOptionalParameterCreator<float> {
 public:
  explicit TreeScalarQuantizationPreprocessedQueryCreator(
      vector<float> inverse_multipliers)
      : inverse_multipliers_(std::move(inverse_multipliers)) {}

  StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
  CreateLeafSearcherOptionalParameters(
      const DatapointPtr<float>& query) const override;

  ConstSpan<float> inverse_multipliers() const;

 private:
  const vector<float> inverse_multipliers_;
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
