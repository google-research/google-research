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

#include "benchmark_utils.h"

#include <utility>

#include "benchmark/benchmark.h"
#include "sketch.h"
#include "utils.h"

namespace sketch {

void Add(benchmark::State& state, Sketch* sketch) {
  for (auto _ : state) {
    state.PauseTiming();
    sketch->Reset();
    auto data = sketch::CreateStream(state.range(0)).first;
    state.ResumeTiming();
    for (const auto& [item, freq] : data) {
      sketch->Add(item, freq);
    }
  }
}

}  // namespace sketch
