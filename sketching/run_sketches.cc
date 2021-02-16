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

// Generate a stream of inputs based on a given distribution, and sketch them.
// Report error statistics, time taken and memory used.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "countmin.h"
#include "frequent.h"
#include "lossy_count.h"
#include "lossy_weight.h"
#include "sketch.h"
#include "utils.h"

DEFINE_int32(stream_size, 1000000, "Number of items in the stream");
DEFINE_int32(lg_stream_range, 20, "Stream elements in 0..2^lg_stream_range");
DEFINE_string(distribution, "zipf", "Which distribution?");
DEFINE_double(zipf_param, 1.1, "Parameter for the Zipf distribution");
DEFINE_double(epsilon, 0.0001, "Heavy hitter fraction");
DEFINE_int32(sketch_size, 100000,
             "Size of sketch, all algorithms get same memory, in bytes");
DEFINE_double(fallback_fraction, 0.2,
              "Fraction of memory used for fallback");
DEFINE_int32(hash_count, 5, "Number of hashes");
DEFINE_int32(frequent_size, 2000, "Items in memory for Frequent (Misra-Gries)");

namespace sketch {

struct SketchStats {
  std::string name;
  ULONG add_time;
  ULONG hh_time;
  std::vector<uint> heavy_hitters;
  uint size;

  float precision;
  float recall;

  float error_mean;
  float error_sd;
  ULONG estimate_time;
};

void TestSketch(float threshold, const std::vector<IntFloatPair>& data,
                const std::vector<float>& counts, Sketch* sketch,
                SketchStats* stats) {
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& kv : data) {
    sketch->Add(kv.first, kv.second);
  }
  auto end_add = std::chrono::high_resolution_clock::now();
  stats->add_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_add - start).count();
  sketch->ReadyToEstimate();
  stats->heavy_hitters = sketch->HeavyHitters(threshold);
  auto end_hh = std::chrono::high_resolution_clock::now();
  stats->hh_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_hh - end_add).count();
  stats->size = sketch->Size();
  double error_sum = 0;
  double error_sq_sum = 0;
  uint stream_range = 1 << absl::GetFlag(FLAGS_lg_stream_range);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < stream_range; ++i) {
    float err = sketch->Estimate(i) - counts[i];
    error_sum += fabs(err);
    error_sq_sum += err * err;
  }
  auto end_est = std::chrono::high_resolution_clock::now();
  stats->estimate_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_est - start).count();
  stats->error_mean = error_sum / stream_range;
  stats->error_sd = sqrt(error_sq_sum / stream_range -
                         stats->error_mean * stats->error_mean);
}

void Evaluate(float threshold, const std::vector<float>& counts,
              int heavy_hitters, SketchStats* stats) {
  float correct = absl::c_count_if(
      stats->heavy_hitters, [&](uint k) { return counts[k] > threshold; });
  stats->precision = static_cast<float>(correct) / stats->heavy_hitters.size();
  stats->recall = static_cast<float>(correct) / heavy_hitters;
}

void PrintOutput(const std::vector<SketchStats>& stats) {
  absl::PrintF(
      "Method\tRecall\tPrec\tSpace\tUpdate Time\tHH time\t"
      "Estimate Err\tEstimate SD\tEstimate time\t\n");
  for (const auto& stat : stats) {
    absl::PrintF("%s\t%0.2f%%\t%0.2f%%\t%d\t%u\t\t%u\t%f\t%f \t%u\n", stat.name,
                 100 * stat.recall, 100 * stat.precision, stat.size,
                 stat.add_time, stat.hh_time, stat.error_mean, stat.error_sd,
                 stat.estimate_time);
  }
}

int DetermineSketchParam(uint max_sketch_size,
                         std::function<uint(uint)> compute_sketch_size) {
  uint lb = 1;
  uint ub = max_sketch_size;
  uint val = max_sketch_size / 8;

  while (ub > lb + 1) {
    uint sketch_size = compute_sketch_size(val);
    if (sketch_size > max_sketch_size) {
      ub = val - 1;
    } else {
      lb = val;
    }
    uint mid = lb + (ub - lb) / 2;  // trick to avoid uint overflow
    val = std::min(std::max(val / 2, mid), val * 2);
  }
  return val;
}

void TestCounts() {
  auto [data, counts] = CreateStream(absl::GetFlag(FLAGS_stream_size),
                                     absl::GetFlag(FLAGS_lg_stream_range),
                                     absl::GetFlag(FLAGS_zipf_param));
  const float threshold =
      absl::GetFlag(FLAGS_epsilon) * absl::GetFlag(FLAGS_stream_size) + 1e-6;
  int heavy_hitters = absl::c_count_if(counts, [&](float c) {
    return c > threshold;
  });
  absl::PrintF("\nStream size: %d, Stream range: 2^%d\n",
               absl::GetFlag(FLAGS_stream_size),
               absl::GetFlag(FLAGS_lg_stream_range));
  absl::PrintF("There were %d elements above threshold %0.2f, for e = %f\n\n",
               heavy_hitters,
               absl::GetFlag(FLAGS_epsilon) * absl::GetFlag(FLAGS_stream_size),
               absl::GetFlag(FLAGS_epsilon));

  std::vector<std::pair<std::string, std::unique_ptr<Sketch>>> sketches;

  const uint cm_hash_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [](uint val) -> uint {
        return CountMin(absl::GetFlag(FLAGS_hash_count), val).Size();
      });
  std::cout << "CM params: hash_size " << cm_hash_size << std::endl;
  sketches.emplace_back(
      "CM", absl::make_unique<CountMin>(
                CountMin(absl::GetFlag(FLAGS_hash_count), cm_hash_size)));

  const uint cmcu_hash_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [](uint val) -> uint {
        return CountMinCU(absl::GetFlag(FLAGS_hash_count), val).Size();
      });
  std::cout << "CM_CU params: hash_size " << cmcu_hash_size << std::endl;
  sketches.emplace_back("CM_CU",
                        absl::make_unique<CountMinCU>(CountMinCU(
                            absl::GetFlag(FLAGS_hash_count), cmcu_hash_size)));

  const uint cmh_hash_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [](uint val) -> uint {
        return CountMinHierarchical(absl::GetFlag(FLAGS_hash_count), val,
                                    absl::GetFlag(FLAGS_lg_stream_range))
            .Size();
      });
  std::cout << "CMH params: hash_size " << cmh_hash_size << std::endl;
  sketches.emplace_back(
      "CMH", absl::make_unique<CountMinHierarchical>(CountMinHierarchical(
                 absl::GetFlag(FLAGS_hash_count), cmh_hash_size,
                 absl::GetFlag(FLAGS_lg_stream_range))));

  const uint cmhcu_hash_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [](uint val) -> uint {
        return CountMinHierarchicalCU(absl::GetFlag(FLAGS_hash_count), val,
                                      absl::GetFlag(FLAGS_lg_stream_range))
            .Size();
      });
  std::cout << "CMH_CU params: hash_size " << cmhcu_hash_size << std::endl;
  sketches.emplace_back(
      "CMH_CU",
      absl::make_unique<CountMinHierarchicalCU>(CountMinHierarchicalCU(
          absl::GetFlag(FLAGS_hash_count), cmhcu_hash_size,
          absl::GetFlag(FLAGS_lg_stream_range))));

  const uint fb_hash_size = DetermineSketchParam(
      static_cast<uint>(absl::GetFlag(FLAGS_sketch_size) *
                        absl::GetFlag(FLAGS_fallback_fraction)),
      [](uint val) -> uint {
        return CountMinCU(absl::GetFlag(FLAGS_hash_count), val).Size();
      });

  const uint lc_window = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size),
      [](uint val) -> uint { return LossyCount(val).Size(); });
  std::cout << "LC params: window_size " << lc_window << std::endl;
  sketches.emplace_back(
      "LC", absl::make_unique<LossyCount>(LossyCount(lc_window)));

  const uint lcfb_window = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [fb_hash_size](uint val) -> uint {
        return LossyCountFallback(val, absl::GetFlag(FLAGS_hash_count),
                                  fb_hash_size)
            .Size();
      });
  std::cout << "LC_FB params: window_size " << lcfb_window
            << " fallback_hashsize " << fb_hash_size << std::endl;
  sketches.emplace_back(
      "LC_FB",
      absl::make_unique<LossyCountFallback>(LossyCountFallback(
          lcfb_window, absl::GetFlag(FLAGS_hash_count), fb_hash_size)));

  const uint lw_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [fb_hash_size](uint val) -> uint {
        return LossyWeight(val, absl::GetFlag(FLAGS_hash_count), fb_hash_size)
            .Size();
      });
  std::cout << "LW params: storage_size " << lw_size
            << " fallback_hashsize " << fb_hash_size << std::endl;
  sketches.emplace_back(
      "LW", absl::make_unique<LossyWeight>(LossyWeight(
                lw_size, absl::GetFlag(FLAGS_hash_count), fb_hash_size)));

  const uint freq_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size),
      [](uint val) -> uint { return Frequent(val).Size(); });
  std::cout << "Freq params: store_size " << freq_size << std::endl;
  sketches.emplace_back(
      "Freq", absl::make_unique<Frequent>(Frequent(freq_size)));

  const uint freqfb_size = DetermineSketchParam(
      absl::GetFlag(FLAGS_sketch_size), [fb_hash_size](uint val) -> uint {
        return FrequentFallback(val, absl::GetFlag(FLAGS_hash_count),
                                fb_hash_size)
            .Size();
      });
  std::cout << "Freq_FB params: store_size " << freqfb_size
            << " fallback_hashsize " << fb_hash_size << std::endl;
  sketches.emplace_back(
      "Freq_FB",
      absl::make_unique<FrequentFallback>(FrequentFallback(
          freqfb_size, absl::GetFlag(FLAGS_hash_count), fb_hash_size)));

  std::cout << std::endl;
  std::vector<SketchStats> sketch_stats;
  for (const auto& [name, sketch] : sketches) {
    SketchStats s;
    s.name = name;
    TestSketch(threshold, data, counts, sketch.get(), &s);
    Evaluate(threshold, counts, heavy_hitters, &s);
    sketch_stats.push_back(s);
  }

  PrintOutput(sketch_stats);
}

}  // namespace sketch

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  sketch::TestCounts();
}
