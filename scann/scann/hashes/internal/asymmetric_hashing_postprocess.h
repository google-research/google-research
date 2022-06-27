// Copyright 2022 The Google Research Authors.
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

#ifndef SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_POSTPROCESS_H_
#define SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_POSTPROCESS_H_

#include <algorithm>
#include <cmath>

#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/prefetch.h"

namespace research_scann {

namespace asymmetric_hashing_internal {

class LimitedInnerFunctor {
 public:
  LimitedInnerFunctor() {}

  LimitedInnerFunctor(double query_norm, ConstSpan<float> norm_inv)
      : norm_inv_(norm_inv),
        query_norm_inv_(query_norm == 0 ? 0 : 1 / query_norm) {}

  float Postprocess(float score, size_t index) const {
    if (ABSL_PREDICT_FALSE(query_norm_inv_ == 0)) {
      return 0.0;
    } else {
      return score * query_norm_inv_ *
             std::min(norm_inv_[index], query_norm_inv_);
    }
  }

 private:
  ConstSpan<float> norm_inv_;
  float query_norm_inv_ = NAN;
};

class AddBiasFunctor {
 public:
  AddBiasFunctor() {}

  explicit AddBiasFunctor(ConstSpan<float> bias, float multiplier)
      : bias_(bias), multiplier_(multiplier) {}

  float Postprocess(float score, size_t index) const {
    return score + bias_[index] * multiplier_;
  }

 private:
  ConstSpan<float> bias_;
  float multiplier_;
};

class IdentityPostprocessFunctor {
 public:
  template <typename T>
  T Postprocess(T score, size_t) const {
    return score;
  }
};

template <typename PostprocessingFunctor>
class ConvertToFloatAndPostprocess {
 public:
  ConvertToFloatAndPostprocess(PostprocessingFunctor postprocess,
                               float inverse_fixed_point_multiplier)
      : postprocess_(postprocess),
        inverse_fixed_point_multiplier_(inverse_fixed_point_multiplier) {}

  template <typename Int>
  float Postprocess(Int score, size_t index) const {
    DCHECK(IsIntegerType<Int>());
    return postprocess_.Postprocess(score * inverse_fixed_point_multiplier_,
                                    index);
  }

 private:
  PostprocessingFunctor postprocess_;
  float inverse_fixed_point_multiplier_;
};

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
