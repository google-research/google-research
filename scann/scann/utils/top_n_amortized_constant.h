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



#ifndef SCANN__UTILS_TOP_N_AMORTIZED_CONSTANT_H_
#define SCANN__UTILS_TOP_N_AMORTIZED_CONSTANT_H_

#include <algorithm>
#include <functional>
#include <utility>

#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace tensorflow {
namespace scann_ops {

template <class T, class Cmp = std::greater<T>>
class TopNAmortizedConstant {
 public:
  virtual ~TopNAmortizedConstant() {}

  explicit TopNAmortizedConstant(size_t limit) : limit_(limit) {
    DCHECK_GT(limit_, 0);
  }

  TopNAmortizedConstant(size_t limit, const Cmp& cmp)
      : limit_(limit), cmp_(cmp) {
    DCHECK_GT(limit_, 0);
  }

  TopNAmortizedConstant() {}

  TopNAmortizedConstant(TopNAmortizedConstant&& rhs) = default;
  TopNAmortizedConstant& operator=(TopNAmortizedConstant&& rhs) = default;

  void push(const T& elem) {
    DCHECK_GT(limit_, 0)
        << "Cannot call Push on uninitialized TopNAmortizedConstant instance.";
    if (ABSL_PREDICT_FALSE(elements_.size() < limit_)) {
      if (ABSL_PREDICT_FALSE(empty() || cmp_(approx_bottom_, elem))) {
        approx_bottom_ = elem;
      }
      elements_.push_back(elem);
    } else if (cmp_(elem, approx_bottom_)) {
      elements_.push_back(elem);
      if (ABSL_PREDICT_FALSE(elements_.size() >= 2 * limit_)) {
        PartitionAndResizeToLimit();
      }
    }
  }

  void OverwriteContents(std::vector<T> new_contents, T approx_bottom) {
    elements_ = std::move(new_contents);
    approx_bottom_ = approx_bottom;
  }

  std::vector<T> TakeUnsorted() {
    DCHECK_GT(limit_, 0) << "Cannot call TakeUnsorted on uninitialized "
                            "TopNAmortizedConstant instance.";
    if (elements_.size() > limit_) PartitionAndResizeToLimit();
    auto result = std::move(elements_);
    elements_.clear();
    return result;
  }

  std::vector<T> Take() {
    DCHECK_GT(limit_, 0) << "Cannot call Take on uninitialized "
                            "TopNAmortizedConstant instance.";
    if (elements_.size() > limit_) PartitionAndResizeToLimit();
    std::sort(elements_.begin(), elements_.end(), cmp_);
    auto result = std::move(elements_);
    elements_.clear();
    return result;
  }

  std::vector<T> Extract() { return Take(); }
  std::vector<T> ExtractUnsorted() { return TakeUnsorted(); }

  const T& approx_bottom() const {
    DCHECK(!elements_.empty());
    return approx_bottom_;
  }

  const T& exact_bottom() {
    DCHECK(!elements_.empty());
    if (elements_.size() > limit_) PartitionAndResizeToLimit();

    return approx_bottom_;
  }

  bool empty() const { return elements_.empty(); }

  size_t size() const { return std::min(limit_, elements_.size()); }

  bool full() const { return elements_.size() >= limit_; }

  size_t limit() const { return limit_; }

  void reserve(size_t n_elements) {
    DCHECK_LE(n_elements, 2 * limit_);
    elements_.reserve(n_elements);
  }

 protected:
  virtual void PartitionElements(std::vector<T>* elements, const Cmp& cmp) {
    std::nth_element(elements->begin(), elements->begin() + limit_ - 1,
                     elements->end(), cmp);
  }

 private:
  void PartitionAndResizeToLimit() {
    DCHECK_GT(elements_.size(), limit_);
    PartitionElements(&elements_, cmp_);
    elements_.resize(limit_);
    approx_bottom_ = elements_.back();
  }

  T approx_bottom_;

  std::vector<T> elements_;

  size_t limit_ = 0;

  Cmp cmp_;

  template <typename Distance>
  friend class TopNeighbors;
};

template <typename Distance>
class TopNeighbors final
    : public TopNAmortizedConstant<pair<DatapointIndex, Distance>,
                                   DistanceComparator> {
 public:
  using Neighbor = pair<DatapointIndex, Distance>;

  TopNeighbors() : TopNAmortizedConstant<Neighbor, DistanceComparator>() {}
  explicit TopNeighbors(size_t num_neighbors)
      : TopNAmortizedConstant<Neighbor, DistanceComparator>(num_neighbors) {}

  void push(DatapointIndex dp_index, Distance distance) {
    this->push(std::make_pair(dp_index, distance));
  }

  using TopNAmortizedConstant<pair<DatapointIndex, Distance>,
                              DistanceComparator>::push;

  template <typename Distance2>
  TopNeighbors<Distance2> CloneWithAlternateDistanceType() const {
    DCHECK(this->empty());
    return TopNeighbors<Distance2>(this->limit());
  }

  template <typename RhsDistance, typename MonotonicTransformation>
  void OverwriteFromClone(TopNeighbors<RhsDistance>* rhs,
                          MonotonicTransformation monotonic_transformation) {
    DCHECK(this->empty());
    DCHECK_EQ(rhs->limit(), this->limit());
    auto rhs_results = rhs->TakeUnsorted();
    this->elements_.resize(rhs_results.size());
    if (!rhs_results.empty()) {
      this->approx_bottom_ =
          std::make_pair(rhs->approx_bottom_.first,
                         monotonic_transformation(rhs->approx_bottom_.second));
    }
    auto dst_ptr = this->elements_.begin();
    auto src_ptr = rhs_results.begin();
    auto src_size = rhs_results.size();
    for (size_t i = 0; i < src_size; ++i) {
      dst_ptr[i].first = src_ptr[i].first;
      dst_ptr[i].second = monotonic_transformation(src_ptr[i].second);
    }
  }

 protected:
  void PartitionElements(std::vector<Neighbor>* elements,
                         const DistanceComparator& cmp) final;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, TopNeighbors);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
