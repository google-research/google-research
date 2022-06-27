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

#include "benchmark_utils.h"

#include <utility>

#include "benchmark/benchmark.h"
#include "sketch.h"
#include "utils.h"

namespace sketch {

constexpr int kAddBatchSize = 32;

std::vector<std::vector<IntFloatPair>> MakeAddStreams(int count) {
  std::vector<std::vector<IntFloatPair>> streams(kAddBatchSize);
  for (auto& stream : streams) {
    stream = CreateStream(count).first;
  }
  return streams;
}

void Add(benchmark::State& state, Sketch* sketch) {
  auto streams = MakeAddStreams(state.range(0));
  while (state.KeepRunningBatch(kAddBatchSize)) {
    for (const auto& stream : streams) {
      for (const auto& [item, freq] : stream) {
        sketch->Add(item, freq);
      }
    }
  }
}

}  // namespace sketch
