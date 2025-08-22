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

#ifndef SCANN_BASE_SEARCH_PARAMETERS_H_
#define SCANN_BASE_SEARCH_PARAMETERS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/base/optimization.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/features.pb.h"
#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class SearcherSpecificOptionalParameters : public VirtualDestructor {};

class SearchParameters {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(SearchParameters);

  SearchParameters() = default;

  SearchParameters(
      int32_t pre_reordering_num_neighbors, float pre_reordering_epsilon,
      int32_t post_reordering_num_neighbors = numeric_limits<int32_t>::max(),
      float post_reordering_epsilon = numeric_limits<float>::infinity())
      : pre_reordering_num_neighbors_(pre_reordering_num_neighbors),
        post_reordering_num_neighbors_(post_reordering_num_neighbors),
        pre_reordering_epsilon_(pre_reordering_epsilon),
        post_reordering_epsilon_(post_reordering_epsilon) {}

  ~SearchParameters() {}

  void SetUnspecifiedParametersFrom(const SearchParameters& defaults);

  Status Validate(bool reordering_enabled) const;

  bool sort_results() const { return sort_results_; }
  void set_sort_results(bool val) { sort_results_ = val; }

  int32_t pre_reordering_num_neighbors() const {
    return pre_reordering_num_neighbors_;
  }
  int32_t post_reordering_num_neighbors() const {
    return post_reordering_num_neighbors_;
  }
  void set_pre_reordering_num_neighbors(int32_t val) {
    pre_reordering_num_neighbors_ = val;
  }
  void set_post_reordering_num_neighbors(int32_t val) {
    post_reordering_num_neighbors_ = val;
  }

  float pre_reordering_epsilon() const { return pre_reordering_epsilon_; }
  float post_reordering_epsilon() const { return post_reordering_epsilon_; }
  void set_pre_reordering_epsilon(float val) { pre_reordering_epsilon_ = val; }
  void set_post_reordering_epsilon(float val) {
    post_reordering_epsilon_ = val;
  }

  int32_t per_crowding_attribute_pre_reordering_num_neighbors() const {
    return per_crowding_attribute_pre_reordering_num_neighbors_;
  }
  void set_per_crowding_attribute_pre_reordering_num_neighbors(int32_t val) {
    per_crowding_attribute_pre_reordering_num_neighbors_ = val;
  }
  int32_t per_crowding_attribute_post_reordering_num_neighbors() const {
    return per_crowding_attribute_post_reordering_num_neighbors_;
  }
  void set_per_crowding_attribute_post_reordering_num_neighbors(int32_t val) {
    per_crowding_attribute_post_reordering_num_neighbors_ = val;
  }

  struct CrowdingDimensionAttributeNumNeighbor {
    std::string dimension;
    std::optional<int64_t> attribute;
    int32_t num_neighbors;
  };
  ConstSpan<CrowdingDimensionAttributeNumNeighbor>
  per_crowding_dimension_attribute_post_reordering_num_neighbors() const {
    return per_crowding_dimension_attribute_post_reordering_num_neighbors_;
  }
  void set_per_crowding_dimension_attribute_post_reordering_num_neighbors(
      ConstSpan<CrowdingDimensionAttributeNumNeighbor> val) {
    per_crowding_dimension_attribute_post_reordering_num_neighbors_ = {
        val.begin(), val.end()};
  }

  bool pre_reordering_crowding_enabled() const {
    return pre_reordering_num_neighbors_ >
           per_crowding_attribute_pre_reordering_num_neighbors_;
  }
  bool post_reordering_crowding_enabled() const {
    return (post_reordering_num_neighbors_ >
            per_crowding_attribute_post_reordering_num_neighbors_) ||
           post_reordering_multi_dimensional_crowding_enabled();
  }

  bool post_reordering_multi_dimensional_crowding_enabled() const {
    for (const auto& entry :
         per_crowding_dimension_attribute_post_reordering_num_neighbors_) {
      if (post_reordering_num_neighbors_ > entry.num_neighbors) return true;
    }
    return false;
  }

  bool crowding_enabled() const {
    return pre_reordering_crowding_enabled() ||
           post_reordering_crowding_enabled();
  }

  bool restricts_enabled() const { return false; }

  const RestrictAllowlist* restrict_whitelist() const { return nullptr; }

  bool IsWhitelisted(DatapointIndex dp_index) const { return false; }

  RestrictAllowlist* mutable_restrict_whitelist() { return nullptr; }

  void EnableRestricts(DatapointIndex database_size, bool default_whitelisted) {
  }

  void DisableRestricts() {}

  const SearcherSpecificOptionalParameters*
  searcher_specific_optional_parameters() const {
    return searcher_specific_optional_parameters_.get();
  }

  void set_searcher_specific_optional_parameters(
      shared_ptr<const SearcherSpecificOptionalParameters> params) {
    searcher_specific_optional_parameters_ = std::move(params);
  }

  template <typename T>
  shared_ptr<const T> searcher_specific_optional_parameters() const {
    return std::dynamic_pointer_cast<const T>(
        searcher_specific_optional_parameters_);
  }

  class UnlockedQueryPreprocessingResults : public VirtualDestructor {};

  void set_unlocked_query_preprocessing_results(
      unique_ptr<UnlockedQueryPreprocessingResults> r) {
    unlocked_query_preprocessing_results_ = std::move(r);
  }

  template <typename Subclass>
  Subclass* unlocked_query_preprocessing_results() const {
    return dynamic_cast<Subclass*>(unlocked_query_preprocessing_results_.get());
  }

  void set_num_random_neighbors(int32_t num_random_neighbors) {
    num_random_neighbors_ = num_random_neighbors;
  }
  int32_t num_random_neighbors() const { return num_random_neighbors_; }

 private:
  bool sort_results_ = true;
  int32_t pre_reordering_num_neighbors_ = -1;
  int32_t post_reordering_num_neighbors_ = -1;
  float pre_reordering_epsilon_ = NAN;
  float post_reordering_epsilon_ = NAN;
  int per_crowding_attribute_pre_reordering_num_neighbors_ =
      numeric_limits<int32_t>::max();
  int per_crowding_attribute_post_reordering_num_neighbors_ =
      numeric_limits<int32_t>::max();
  std::vector<CrowdingDimensionAttributeNumNeighbor>
      per_crowding_dimension_attribute_post_reordering_num_neighbors_;
  int32_t num_random_neighbors_ = 0;

  shared_ptr<const SearcherSpecificOptionalParameters>
      searcher_specific_optional_parameters_;

  unique_ptr<UnlockedQueryPreprocessingResults>
      unlocked_query_preprocessing_results_;
};

}  // namespace research_scann

#endif
