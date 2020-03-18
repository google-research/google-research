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
#include "absl/memory/memory.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/random/zipf_distribution.h"
#include "countmin.h"
#include "frequent.h"
#include "lossy_count.h"
#include "lossy_weight.h"
#include "sketch.h"
#include "utils.h"

DEFINE_int32(stream_size, 1000000, "Number of items in the stream");
DEFINE_int32(lg_stream_range, 20, "Stream elements in 0..2^log_stream_range");
DEFINE_string(distribution, "zipf", "Which distribution?");
DEFINE_double(zipf_param, 1.1, "Parameter for the Zipf distribution");
DEFINE_double(epsilon, 0.0001, "Heavy hitter fraction");
DEFINE_int32(hash_count, 5, "Number of hashes");
DEFINE_int32(hash_size, 2048, "Size of each hash");
DEFINE_int32(frequent_size, 2000, "Items in memory for Frequent (Misra-Gries)");

namespace sketch {

std::pair<std::vector<IntFloatPair>, std::vector<float>> CreateStream() {
  std::vector<IntFloatPair> data;
  uint stream_range = 1 << FLAGS_lg_stream_range;
  std::vector<float> counts(stream_range, 0.0);
  BitGenerator bit_gen;
  absl::BitGenRef& gen = *bit_gen.BitGen();
  absl::zipf_distribution<uint> zipf(stream_range, FLAGS_zipf_param, 1.0);
  ULONG a = absl::uniform_int_distribution<int>(0, stream_range - 1)(gen);
  ULONG b = absl::uniform_int_distribution<int>(0, stream_range - 1)(gen);
  for (int i = 0; i < FLAGS_stream_size; ++i) {
    const auto& k = Hash(a, b, zipf(gen), stream_range);
    counts.at(k) += 1.0;
    data.emplace_back(k, 1.0);
  }
  return {data, counts};
}

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
  uint stream_range = 1 << FLAGS_lg_stream_range;
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
  printf("Method\tRecall\tPrec\tSpace\tUpdate Time\tHH time\t"
         "Estimate Err\tEstimate SD\tEstimate time\t\n");
  for (const auto& stat : stats) {
    printf("%s\t%0.2f%%\t%0.2f%%\t%d\t%llu\t\t%llu\t%f\t%f \t%llu\n",
           stat.name.c_str(),
           100 * stat.recall, 100 * stat.precision, stat.size, stat.add_time,
           stat.hh_time, stat.error_mean, stat.error_sd, stat.estimate_time);
  }
}

void TestCounts() {
  auto [data, counts] = CreateStream();
  const float threshold = FLAGS_epsilon * FLAGS_stream_size + 1e-6;
  int heavy_hitters = absl::c_count_if(counts, [&](float c) {
    return c > threshold;
  });
  printf("\nStream size: %d, Stream range: 2^%d\n", FLAGS_stream_size,
         FLAGS_lg_stream_range);
  printf("There were %d elements above threshold %0.2f, for e = %f\n\n",
         heavy_hitters, FLAGS_epsilon * FLAGS_stream_size, FLAGS_epsilon);

  std::vector<std::pair<std::string, std::unique_ptr<Sketch>>> sketches;

  sketches.emplace_back(
      "CM", absl::make_unique<CountMin>(
          CountMin(FLAGS_hash_count, FLAGS_hash_size)));
  sketches.emplace_back(
      "CM_CU", absl::make_unique<CountMinCU>(
          CountMinCU(FLAGS_hash_count, FLAGS_hash_size)));
  sketches.emplace_back(
      "LC", absl::make_unique<LossyCount>(LossyCount(
          (int)(1.0 / FLAGS_epsilon))));
  sketches.emplace_back(
      "LC_FB", absl::make_unique<LossyCount_Fallback>(LossyCount_Fallback(
          (int)(1.0 / FLAGS_epsilon), FLAGS_hash_count, FLAGS_hash_size)));
  sketches.emplace_back(
      "LW", absl::make_unique<LossyWeight>(LossyWeight(
          FLAGS_frequent_size, FLAGS_hash_count, FLAGS_hash_size)));
  sketches.emplace_back(
      "Freq", absl::make_unique<Frequent>(Frequent(FLAGS_frequent_size)));
  sketches.emplace_back(
      "Freq_FB", absl::make_unique<Frequent_Fallback>(Frequent_Fallback(
      FLAGS_frequent_size, FLAGS_hash_count, FLAGS_hash_size)));
  sketches.emplace_back(
      "CMH", absl::make_unique<CountMinHierarchical>(CountMinHierarchical(
          FLAGS_hash_count, FLAGS_hash_size, FLAGS_lg_stream_range)));
  sketches.emplace_back(
      "CMH_CU", absl::make_unique<CountMinHierarchicalCU>(
          CountMinHierarchicalCU(
              FLAGS_hash_count, FLAGS_hash_size, FLAGS_lg_stream_range)));

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
