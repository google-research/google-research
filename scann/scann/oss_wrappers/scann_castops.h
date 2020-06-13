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

#ifndef SCANN__OSS_WRAPPERS_SCANN_CASTOPS_H_
#define SCANN__OSS_WRAPPERS_SCANN_CASTOPS_H_

#include <limits>

namespace tensorflow {
namespace scann_ops {
namespace cast_ops {

inline float DoubleToFloat(double value) {
  if (value < std::numeric_limits<float>::lowest())
    return -std::numeric_limits<float>::infinity();
  if (value > std::numeric_limits<float>::max())
    return std::numeric_limits<float>::infinity();

  return static_cast<float>(value);
}

inline float DoubleToFiniteFloat(double value) {
  if (value < std::numeric_limits<float>::lowest())
    return std::numeric_limits<float>::lowest();
  if (value > std::numeric_limits<float>::max())
    return std::numeric_limits<float>::max();

  return static_cast<float>(value);
}

}  // namespace cast_ops
}  // namespace scann_ops
}  // namespace tensorflow

#endif
