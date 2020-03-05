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

#include "experiment_util.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>

#include "task.proto.h"
#include "google/protobuf/repeated_field.h"
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"

namespace automl_zero {

using ::absl::make_unique;  // NOLINT
using ::proto2::RepeatedField;
using ::std::endl;           // NOLINT
using ::std::equal;          // NOLINT
using ::std::function;       // NOLINT
using ::std::set;            // NOLINT
using ::std::unique_ptr;     // NOLINT
using ::std::unordered_map;  // NOLINT
using ::std::unordered_set;  // NOLINT
using ::std::vector;         // NOLINT

vector<Op> ExtractOps(const RepeatedField<int>& ops_src) {
  vector<Op> ops_dest;
  for (IntegerT i = 0; i < ops_src.size(); ++i) {
    ops_dest.push_back(static_cast<Op>(ops_src.Get(i)));
  }
  return ops_dest;
}

}  // namespace automl_zero
