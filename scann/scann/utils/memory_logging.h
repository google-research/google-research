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



#ifndef SCANN__UTILS_MEMORY_LOGGING_H_
#define SCANN__UTILS_MEMORY_LOGGING_H_

#include <hash_set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "scann/data_format/datapoint.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

static const double bytes_in_mb = 1024.0 * 1024.0;

constexpr double kBytesInMb = 1024.0 * 1024.0;

template <typename T>
size_t VectorStorage(const vector<T>& vec) {
  return sizeof(vec) + vec.capacity() * sizeof(vec[0]);
}

template <typename T>
size_t HashSetStorage(const std::unordered_set<T>& set) {
  return sizeof(set) + set.bucket_count() * sizeof(void*) +
         set.size() * sizeof(pair<T, void*>);
}

template <typename T, typename U, typename Hash>
size_t HashMapStorage(const std::unordered_map<T, U, Hash>& map) {
  return sizeof(map) + map.bucket_count() * sizeof(void*) +
         map.size() * sizeof(std::tuple<T, U, void*>);
}

template <typename K, typename V, typename... Args>
size_t HashMapStorage(const flat_hash_map<K, V, Args...>& map) {
  return sizeof(map) + map.bucket_count() * (sizeof(pair<K, V>) + 1);
}

template <typename T>
size_t HashSetStorage(const flat_hash_set<T>& set) {
  return sizeof(set) + set.bucket_count() * (sizeof(T) + 1);
}

template <typename Vector>
size_t TwoDimensionalVectorStorage(const Vector& vec) {
  size_t result = sizeof(vec) + vec.capacity() * sizeof(vec[0]);
  for (const auto& elem : vec) {
    result += elem.capacity() * sizeof(elem[0]);
  }

  return result;
}

size_t GfvStorage(const GenericFeatureVector& gfv);

size_t GfvStorage(const vector<GenericFeatureVector>& gfvs);

template <typename T>
size_t DatapointStorage(const Datapoint<T>& dp) {
  return sizeof(dp) + VectorStorage(dp.indices()) + VectorStorage(dp.values());
}

template <typename T>
size_t DatapointStorage(const vector<Datapoint<T> >& dps) {
  size_t result = sizeof(dps);
  for (const auto& dp : dps) {
    result += DatapointStorage(dp);
  }

  return result;
}

std::string GetTcMallocLogString();

#define SCANN_LOG_TCMALLOC() LOG(INFO) << GetTcMallocLogString();
#define SCANN_VLOG_TCMALLOC(verbosity_level) \
  VLOG(verbosity_level) << GetTcMallocLogString();

inline void LogTcMalloc(int verbosity_level = 0) {
  VLOG(verbosity_level) << GetTcMallocLogString();
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
