// Copyright 2021 The Google Research Authors.
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

#ifndef FAIRNESS_AND_BIAS_SELECTION_UTILS_H_
#define FAIRNESS_AND_BIAS_SELECTION_UTILS_H_

#include <string>
#include <vector>

namespace fair_secretary {

// An element for the Secretaty problem.
struct SecretaryInstance {
  SecretaryInstance(double value_, int color_, int type_ = 0) {
    value = value_;
    color = color_;
    type = type_;
  }
  // The value of the element.
  double value;
  // The color class that it belongs to.
  int color;
  // The type of the element.
  int type;
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_UTILS_H_
