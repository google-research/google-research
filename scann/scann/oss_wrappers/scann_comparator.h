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

#ifndef SCANN__OSS_WRAPPERS_SCANN_COMPARATOR_H_
#define SCANN__OSS_WRAPPERS_SCANN_COMPARATOR_H_

#include <utility>

namespace tensorflow {
namespace scann_ops {
namespace internal {

struct OrderBySecond {
  template <typename First, typename Second>
  bool operator()(const std::pair<First, Second>& a,
                  const std::pair<First, Second>& b) const {
    if (a.second < b.second) return true;
    if (a.second > b.second) return false;
    return a.first < b.first;
  }
};

}  // namespace internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
