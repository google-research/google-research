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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TEST_UTIL_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TEST_UTIL_H_

#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "definitions.h"
#include "absl/container/node_hash_set.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "util/gtl/map_util.h"

namespace brain {
namespace evolution {
namespace amlz {

namespace internal {

template <class PrintableTypeT>
std::string AsString(PrintableTypeT input) {
  ::std::stringstream stream;
  stream << input;
  return stream.str();
}

template <>
inline std::string AsString<uint8_t>(uint8_t input) {
  ::std::stringstream stream;
  stream << static_cast<IntegerT>(input);
  return stream.str();
}

template <>
inline std::string AsString<uint16_t>(uint16_t input) {
  ::std::stringstream stream;
  stream << static_cast<IntegerT>(input);
  return stream.str();
}

template <class PrintableTypeT>
std::string AsString(const std::pair<PrintableTypeT, PrintableTypeT>& input) {
  ::std::stringstream stream;
  stream << "(" << AsString(input.first) << ","
         << AsString(input.second) << ")";
  return stream.str();
}

template <class PrintableTypeT>
std::string AsString(absl::node_hash_set<PrintableTypeT> input) {
  ::std::stringstream stream;
  stream << "{";
  for (auto it = input.begin(); it != input.end(); ++it) {
    stream << AsString(*it) << ",";
  }
  stream << "}";
  return stream.str();
}

}  // namespace internal

// Returns whether a given function produces all values in a required set while
// only producing values within an allowed set.
template <class HashableType>
bool IsEventually(std::function<HashableType(void)> func,
                  // Only these values are allowed or returns false. If allowed
                  // is empty, all possible values are allowed.
                  const absl::node_hash_set<HashableType>& allowed,
                  // Each of these values must be produced at least once, or
                  // returns false. Must be non-empty.
                  const absl::node_hash_set<HashableType>& required,
                  // The maximum amount of time to search. If it reaches this
                  // limit, returns false.
                  double max_secs = 3.0) {
  CHECK(!required.empty());
  if (!allowed.empty() && allowed.size() < required.size()) {
    LOG(INFO) << "Allowed set must be larger than or equal to required set, "
              << "or must be empty." << std::endl;
    return false;
  }
  absl::Time start_time = absl::Now();
  absl::node_hash_set<HashableType> missing = required;
  while ((absl::Now() - start_time) < absl::Seconds(max_secs)) {
    HashableType result = func();
    if (!allowed.empty() && allowed.find(result) == allowed.end()) {
      LOG(INFO) << "Found disallowed value: " << internal::AsString(result)
                << std::endl;
      LOG(INFO) << "Allowed values are: " << internal::AsString(allowed)
                << std::endl;
      return false;
    }
    auto missing_it = missing.find(result);
    if (missing_it != missing.end()) {
      missing.erase(missing_it);
    }
    if (missing.empty()) {
      return true;
    }
  }
  LOG(INFO) << "Missing values: " << internal::AsString(missing) << std::endl;
  return false;
}

// Checks that a function never produces a value in a given set.
template <class HashableType>
bool IsNever(
    // The function in question.
    std::function<HashableType(void)> func,
    // Set of values that cannot be produced. Must be non-empty.
    const absl::node_hash_set<HashableType>& excluded,
    // The amount of time to search.
    double max_secs = 3.0) {
  CHECK(!excluded.empty());
  clock_t start_time = clock();
  while (clock() - start_time < max_secs * 1000.0) {
    HashableType result = func();
    if (excluded.find(result) != excluded.end()) {
      LOG(INFO) << "Found excluded value: " << internal::AsString(result)
                << std::endl;
      return false;
    }
  }
  return true;
}

template <class DiscreteSortableType>
std::map<DiscreteSortableType, IntegerT> ComputeHistogram(
    std::function<DiscreteSortableType(void)> func,
    const IntegerT num_samples) {
  std::map<DiscreteSortableType, IntegerT> histogram;
  for (IntegerT sample = 0; sample < num_samples; ++sample) {
    DiscreteSortableType key = func();
    ++gtl::LookupOrInsert(&histogram, key, 0);
  }
  return histogram;
}

template <class DiscreteSortableType>
void PrintDistribution(
    std::function<DiscreteSortableType(void)> func,
    const IntegerT num_samples) {
  std::map<DiscreteSortableType, IntegerT> histogram =
      ComputeHistogram(func, num_samples);
  for (auto const& kv : histogram) {
    std::cout << "Class " << internal::AsString(kv.first) << ": " << kv.second
              << std::endl;
  }
}

template <class NumberT>
absl::node_hash_set<NumberT> Range(const NumberT first, const NumberT last) {
  absl::node_hash_set<NumberT> generated;
  for (NumberT next = first; next < last; ++next) {
    generated.insert(next);
  }
  return generated;
}

template <class NumberT>
absl::node_hash_set<std::pair<NumberT, NumberT>> CartesianProduct(
    const absl::node_hash_set<NumberT>& set1,
    const absl::node_hash_set<NumberT>& set2) {
  absl::node_hash_set<std::pair<NumberT, NumberT>> product;
  for (const NumberT number1 : set1) {
    for (const NumberT number2 : set2) {
      product.insert(std::make_pair(number1, number2));
    }
  }
  return product;
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TEST_UTIL_H_
