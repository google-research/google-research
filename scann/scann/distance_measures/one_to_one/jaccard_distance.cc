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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/distance_measures/one_to_one/jaccard_distance.h"

namespace tensorflow {
namespace scann_ops {

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(GeneralJaccardDistance,
                                              kEarlyStoppingNotSupported);
SCANN_REGISTER_DISTANCE_MEASURE(GeneralJaccardDistance);

double BinaryJaccardDistance::GetDistanceDense(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());

  int32_t xor_sum = 0;
  int32_t or_sum = 0;
  size_t i = 0;
  for (; i + 8 <= a.nonzero_entries(); i += 8) {
    const unsigned long long_t aval = ABSL_INTERNAL_UNALIGNED_LOAD64(a.values() + i);
    const unsigned long long_t bval = ABSL_INTERNAL_UNALIGNED_LOAD64(b.values() + i);
    xor_sum += bits::CountOnes64(aval ^ bval);
    or_sum += bits::CountOnes64(aval | bval);
  }

  if (i + 4 <= a.nonzero_entries()) {
    const uint32_t aval = ABSL_INTERNAL_UNALIGNED_LOAD32(a.values() + i);
    const uint32_t bval = ABSL_INTERNAL_UNALIGNED_LOAD32(b.values() + i);
    xor_sum += bits::CountOnes64(aval ^ bval);
    or_sum += bits::CountOnes64(aval | bval);
    i += 4;
  }

  if (i + 2 <= a.nonzero_entries()) {
    const uint32_t aval = ABSL_INTERNAL_UNALIGNED_LOAD16(a.values() + i);
    const uint32_t bval = ABSL_INTERNAL_UNALIGNED_LOAD16(b.values() + i);
    xor_sum += bits::CountOnes64(aval ^ bval);
    or_sum += bits::CountOnes64(aval | bval);
    i += 2;
  }

  if (i < a.nonzero_entries()) {
    const uint32_t aval = a.values()[i];
    const uint32_t bval = b.values()[i];
    xor_sum += bits::CountOnes64(aval ^ bval);
    or_sum += bits::CountOnes64(aval | bval);
  }

  return (ABSL_PREDICT_FALSE(or_sum == 0))
             ? 1.0
             : (static_cast<double>(xor_sum) / static_cast<double>(or_sum));
}

double BinaryJaccardDistance::GetDistanceSparse(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  if (a.nonzero_entries() == 0 || b.nonzero_entries() == 0) {
    return 1;
  }
  int32_t intersection = 0;
  int32_t sum = 0;
  size_t a_front = 0, b_front = 0;
  size_t a_back = a.nonzero_entries() - 1, b_back = b.nonzero_entries() - 1;
  while (a_front < a_back && b_front < b_back) {
    const size_t to_add_front1 = a.indices()[a_front] <= b.indices()[b_front];
    const size_t to_add_front2 = a.indices()[a_front] >= b.indices()[b_front];
    const size_t to_sub_back2 = a.indices()[a_back] <= b.indices()[b_back];
    const size_t to_sub_back1 = a.indices()[a_back] >= b.indices()[b_back];
    intersection += (a.indices()[a_front] == b.indices()[b_front]);
    intersection += (a.indices()[a_back] == b.indices()[b_back]);
    a_front += to_add_front1;
    b_front += to_add_front2;
    a_back -= to_sub_back1;
    b_back -= to_sub_back2;
  }
  if (a_front == a_back) {
    for (; b_front <= b_back; ++b_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[a_front] == b.indices()[b_front])) {
        intersection++;
        break;
      }
    }
  } else if (b_front == b_back) {
    for (; a_front <= a_back; ++a_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[a_front] == b.indices()[b_front])) {
        intersection++;
        break;
      }
    }
  }
  sum = a.nonzero_entries() + b.nonzero_entries() - intersection;
  return 1 - (static_cast<double>(intersection) / static_cast<double>(sum));
}

SCANN_REGISTER_DISTANCE_MEASURE(BinaryJaccardDistance)

}  // namespace scann_ops
}  // namespace tensorflow
