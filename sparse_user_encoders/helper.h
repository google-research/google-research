// Copyright 2025 The Google Research Authors.
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

#ifndef HELPER_H
#define HELPER_H

// NOLINTBEGIN
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "recommender.h"
// NOLINTEND

template <typename T, typename F>
void parallel_iterate_over_map(const std::unordered_map<int, T>& map,
                               F work_per_value, int num_threads) {
  std::mutex m;
  auto index_iter = map.begin();   // protected by m
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread([&](const int thread_index){
      while (true) {
        // Get a new entry to work on.
        m.lock();
        if (index_iter == map.end()) {
          m.unlock();
          return;
        }
        const int key = index_iter->first;
        const T& value = index_iter->second;
        // Move to the next entry
        ++index_iter;
        m.unlock();

        work_per_value(thread_index, value, key, &m);
      }
    }, i));
  }
  // Join all threads.
  for (auto& thread : threads) {
    thread.join();
  }
}

class Timer {
 public:
  Timer() {
    time_start_ = std::chrono::steady_clock::now();
  }
  int timeSinceStartInMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - time_start_).count();
  }
  std::string timeSinceStartAsString() {
    int time_in_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - time_start_).count();
    int time_in_s = time_in_ms / 1000;
    int time_in_m = time_in_s / 60;
    float time_in_h = time_in_m / 60.0;

    std::ostringstream stringStream;
    stringStream
        << time_in_s << " sec / "
        << time_in_m << " min / "
        << time_in_h << " hrs";
    return stringStream.str();
  }

 private:
  std::chrono::steady_clock::time_point time_start_;
};

class Flags {
 public:
  Flags(int argc, char* argv[]) {
    // Parse flags. This is a simple implementation to avoid external
    // dependencies.
    for (int i = 1; i < argc; ++i) {
      assert(i < (argc-1));
      std::string flag_name = argv[i];
      assert(flag_name.at(0) == '-');
      if (flag_name.at(1) == '-') {
        flag_name = flag_name.substr(2);
      } else {
        flag_name = flag_name.substr(1);
      }
      ++i;
      std::string flag_value = argv[i];
      flags_[flag_name] = flag_value;
    }
  }
  void setDefault(const std::string& flag_name, const std::string& value) {
    if (!hasFlag(flag_name)) { flags_[flag_name] = value; }
  }
  const bool hasFlag(const std::string& flag_name) {
    return flags_.count(flag_name) >= 1;
  }
  const std::string getStrValue(const std::string& flag_name) {
    return flags_.at(flag_name);
  }
  const int getIntValue(const std::string& flag_name) {
    return std::atoi(flags_.at(flag_name).c_str());  // NOLINT
  }
  const float getFloatValue(const std::string& flag_name) {
    return std::atof(flags_.at(flag_name).c_str());  // NOLINT
  }
  const std::vector<std::string> getStrValues(const std::string& flag_name) {
    std::vector<std::string> result;
    std::stringstream ss(getStrValue(flag_name));
    std::string temp;
    while (std::getline(ss, temp, ',')) {
      result.push_back(temp);
    }
    return result;
  }
  const std::vector<int> getIntValues(const std::string& flag_name) {
    std::vector<int> result;
    for (const std::string& s : getStrValues(flag_name)) {
      result.push_back(std::atoi(s.c_str()));  // NOLINT
    }
    return result;
  }
  const std::vector<float> getFloatValues(const std::string& flag_name) {
    std::vector<float> result;
    for (const std::string& s : getStrValues(flag_name)) {
      result.push_back(std::atof(s.c_str()));  // NOLINT
    }
    return result;
  }

 private:
  std::unordered_map<std::string, std::string> flags_;
};

#endif  // HELPER_H
