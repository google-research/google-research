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

#include "sliding_window_framework.h"

namespace sliding_window {

void SummaryAlg::process_point(const int64_t time,
                               const std::vector<double>& point) {
  if (is_empty_) {
    is_empty_ = false;
    first_element_time_ = time;
  }
  process_point_impl(time, point);
}

// Resets the summary.
void SummaryAlg::reset() {
  is_empty_ = true;
  first_element_time_ = -1;
  reset_impl();
}

}  //  namespace sliding_window
