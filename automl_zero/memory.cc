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

#include "memory.h"

namespace automl_zero {

// Define here all the class-instances of the template that will be compiled.
// Bigger than 32 leads to allocs too large for dense-storage in Eigen in stack.
template class Memory<2>;
template class Memory<4>;
template class Memory<8>;
template class Memory<16>;
template class Memory<32>;

}  // namespace automl_zero
