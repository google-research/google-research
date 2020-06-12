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

#include "scann/utils/alignment.h"

#include "scann/utils/common.h"

namespace tensorflow {
namespace scann_ops {

AlignedBuffer MakeCacheAlignedCopy(ConstSpan<uint8_t> span) {
  const size_t padded_size1 = NextMultipleOf(span.size(), 64);

  const size_t padded_size2 = padded_size1 + 63;

  auto padded = make_unique<uint8_t[]>(padded_size2);
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
      NextMultipleOf(reinterpret_cast<uintptr_t>(padded.get()), 64));
  CHECK_LE(ptr + NextMultipleOf(span.size(), 64), padded.get() + padded_size2);

  std::copy(span.begin(), span.end(), ptr);

  std::fill(ptr + span.size(), ptr + padded_size1, 0xFF);
  return {std::move(padded), ptr};
}

}  // namespace scann_ops
}  // namespace tensorflow
