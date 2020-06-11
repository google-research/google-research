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



#include "scann/utils/util_functions.h"

#include <numeric>

#include "absl/base/internal/sysinfo.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace scann_ops {

#ifdef __SSE3__

float MaxAbsValue(ConstSpan<float> arr) {
  size_t num_simd_iter = arr.size() / 4;
  const float* ptr = arr.data();
  __m128 accumulators = _mm_setzero_ps();
  constexpr int32_t abs_mask_scalar = 0x7FFFFFFF;
  static const __m128 abs_mask_vector =
      _mm_castsi128_ps(_mm_set1_epi32(abs_mask_scalar));

  float result = 0.0f;
  if (num_simd_iter != 0) {
    for (; num_simd_iter != 0; --num_simd_iter, ptr += 4) {
      __m128 vals = _mm_loadu_ps(ptr);
      __m128 abs_vals = _mm_and_ps(vals, abs_mask_vector);
      accumulators = _mm_max_ps(accumulators, abs_vals);
    }

    accumulators = _mm_max_ps(
        accumulators, _mm_shuffle_ps(accumulators, accumulators, 0b1110));
    result = std::max(accumulators[0], accumulators[1]);
  }

  const float* end = arr.data() + arr.size();
  for (; ptr < end; ++ptr) {
    result = std::max(result, std::abs(*ptr));
  }

  return result;
}

#else

float MaxAbsValue(ConstSpan<float> arr) {
  float max_abs_value = 0.0f;
  for (float elem : arr) {
    max_abs_value = std::max(max_abs_value, std::abs(elem));
  }

  return max_abs_value;
}

#endif

void RemoveNeighborsPastLimit(DatapointIndex num_neighbors,
                              NNResultsVector* result) {
  DCHECK(result);
  if (num_neighbors == 0) {
    result->clear();
    return;
  }
  if (result->size() > num_neighbors) {
    ZipNthElementBranchOptimized(DistanceComparatorBranchOptimized(),
                                 num_neighbors - 1, result->begin(),
                                 result->end());
    result->resize(num_neighbors);
  }
}

namespace {

struct PartiallyConsumedNeighborList {
  protobuf::RepeatedPtrField<NearestNeighbors::Neighbor>* neighbor_list =
      nullptr;

  int pos = 0;
};

class PartiallyConsumedNeighborListComparator {
 public:
  bool operator()(const PartiallyConsumedNeighborList& a,
                  const PartiallyConsumedNeighborList& b) const {
    DistanceComparator comp;
    return comp(b.neighbor_list->Get(b.pos), a.neighbor_list->Get(a.pos));
  }
};

void ReleaseAllNeighbors(
    protobuf::RepeatedPtrField<NearestNeighbors::Neighbor>* list) {
  while (list->size()) {
    list->ReleaseLast();
  }
}

template <typename Lambda>
inline NearestNeighbors MergeNeighborListsImpl(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors,
    Lambda should_drop) {
  DCHECK(neighbor_lists.size());
  NearestNeighbors result;

  if (neighbor_lists.size() == 1) {
    result.Swap(&neighbor_lists[0]);
    return result;
  }

  vector<PartiallyConsumedNeighborList> heap;
  heap.reserve(neighbor_lists.size());

  int total_neighbors = 0;
  for (auto& list : neighbor_lists) {
    total_neighbors += list.neighbor_size();
    PartiallyConsumedNeighborList pc_list;
    if (list.neighbor_size() > 0) {
      pc_list.neighbor_list = list.mutable_neighbor();
      heap.push_back(pc_list);
    }
    if (list.has_metadata()) {
      DCHECK(!result.has_metadata())
          << "Multiple shards set metadata for the query!";
      result.set_metadata(list.metadata());
    }
  }

  result.mutable_docid()->swap(*neighbor_lists[0].mutable_docid());
  if (heap.size() == 0) {
    return result;
  }

  PartiallyConsumedNeighborListComparator comp;
  std::make_heap(heap.begin(), heap.end(), comp);
  result.mutable_neighbor()->Reserve(std::min(total_neighbors, num_neighbors));
  while (result.neighbor_size() < num_neighbors && heap.size() > 1) {
    std::pop_heap(heap.begin(), heap.end(), comp);

    auto merge_from = &heap.back();
    auto next_neighbor = merge_from->neighbor_list->Mutable(merge_from->pos++);

    if (should_drop(next_neighbor)) {
      delete next_neighbor;
    } else {
      result.mutable_neighbor()->AddAllocated(next_neighbor);
    }

    if (merge_from->pos < merge_from->neighbor_list->size()) {
      std::push_heap(heap.begin(), heap.end(), comp);
    } else {
      ReleaseAllNeighbors(heap.back().neighbor_list);
      heap.pop_back();
    }
  }

  auto last_list = &heap.front();
  while (result.neighbor_size() < num_neighbors &&
         last_list->pos < last_list->neighbor_list->size()) {
    auto next_neighbor = last_list->neighbor_list->Mutable(last_list->pos++);

    if (should_drop(next_neighbor)) {
      delete next_neighbor;
    } else {
      result.mutable_neighbor()->AddAllocated(next_neighbor);
    }
  }

  for (auto& list : heap) {
    while (list.neighbor_list->size() > list.pos) {
      list.neighbor_list->RemoveLast();
    }

    ReleaseAllNeighbors(list.neighbor_list);
  }

  for (auto& list : neighbor_lists) {
    list.Clear();
  }

  return result;
}

template <typename Lambda>
void MergeNeighborListsSwapImpl(MutableSpan<NearestNeighbors*> neighbor_lists,
                                int num_neighbors, Lambda should_drop,
                                NearestNeighbors* result) {
  DCHECK(neighbor_lists.size());
  DCHECK(result);
  result->Clear();

  if (neighbor_lists.size() == 1) {
    result->Swap(neighbor_lists[0]);
    return;
  }

  vector<PartiallyConsumedNeighborList> heap;
  heap.reserve(neighbor_lists.size());

  int total_neighbors = 0;
  for (NearestNeighbors* list : neighbor_lists) {
    total_neighbors += list->neighbor_size();
    PartiallyConsumedNeighborList pc_list;
    if (list->neighbor_size() > 0) {
      pc_list.neighbor_list = list->mutable_neighbor();
      heap.push_back(pc_list);
    }
  }

  result->mutable_docid()->swap(*neighbor_lists[0]->mutable_docid());
  if (heap.size() == 0) {
    return;
  }

  PartiallyConsumedNeighborListComparator comp;
  std::make_heap(heap.begin(), heap.end(), comp);
  result->mutable_neighbor()->Reserve(std::min(total_neighbors, num_neighbors));
  while (result->neighbor_size() < num_neighbors && heap.size() > 1) {
    std::pop_heap(heap.begin(), heap.end(), comp);

    auto merge_from = &heap.back();
    auto next_neighbor = merge_from->neighbor_list->Mutable(merge_from->pos++);
    if (!should_drop(next_neighbor)) {
      result->add_neighbor()->Swap(next_neighbor);
    }

    if (merge_from->pos < merge_from->neighbor_list->size()) {
      std::push_heap(heap.begin(), heap.end(), comp);
    } else {
      heap.pop_back();
    }
  }

  auto last_list = &heap.front();
  while (result->neighbor_size() < num_neighbors &&
         last_list->pos < last_list->neighbor_list->size()) {
    auto next_neighbor = last_list->neighbor_list->Mutable(last_list->pos++);
    if (!should_drop(next_neighbor)) {
      result->add_neighbor()->Swap(next_neighbor);
    }
  }
}

}  // namespace

NearestNeighbors MergeNeighborLists(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors) {
  auto return_false = [](const NearestNeighbors::Neighbor* nn) {
    return false;
  };

  return MergeNeighborListsImpl(neighbor_lists, num_neighbors, return_false);
}

NearestNeighbors MergeNeighborListsWithCrowding(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors,
    int per_crowding_attribute_num_neighbors) {
  std::unordered_map<int64_t, int> crowding_counts;
  auto is_crowded_out = [&crowding_counts,
                         per_crowding_attribute_num_neighbors](
                            const NearestNeighbors::Neighbor* nn) {
    const int crowding_count = ++crowding_counts[nn->crowding_attribute()];
    return crowding_count > per_crowding_attribute_num_neighbors;
  };

  return MergeNeighborListsImpl(neighbor_lists, num_neighbors, is_crowded_out);
}

NearestNeighbors MergeNeighborListsRemoveDuplicateDocids(
    MutableSpan<NearestNeighbors> neighbor_lists, int num_neighbors) {
  std::unordered_set<std::string> prev_docids;
  auto docid_seen = [&prev_docids](const NearestNeighbors::Neighbor* nn) {
    auto iter = prev_docids.find(nn->docid());
    if (iter == prev_docids.end()) {
      prev_docids.insert(nn->docid());
      return false;
    }

    return true;
  };

  return MergeNeighborListsImpl(neighbor_lists, num_neighbors, docid_seen);
}

void MergeNeighborListsSwap(MutableSpan<NearestNeighbors*> neighbor_lists,
                            int num_neighbors, NearestNeighbors* result) {
  return MergeNeighborListsSwapImpl(
      neighbor_lists, num_neighbors,
      [](NearestNeighbors::Neighbor* nn) { return false; }, result);
}

void MergeNeighborListsWithCrowdingSwap(
    MutableSpan<NearestNeighbors*> neighbor_lists, int num_neighbors,
    int per_crowding_attribute_num_neighbors, NearestNeighbors* result) {
  std::unordered_map<int64_t, int> crowding_counts;
  auto is_crowded_out = [&crowding_counts,
                         per_crowding_attribute_num_neighbors](
                            const NearestNeighbors::Neighbor* nn) {
    const int crowding_count = ++crowding_counts[nn->crowding_attribute()];
    return crowding_count > per_crowding_attribute_num_neighbors;
  };
  MergeNeighborListsSwapImpl(neighbor_lists, num_neighbors, is_crowded_out,
                             result);
}

void LogCPUInfo() {
  LOG(INFO) << "CPU Info:\n"
            << "Model number:  " << port::CPUModelNum() << "\n"
            << "Clock speed:  " << port::NominalCPUFrequency() / 1.0e9
            << " GHz\n"
            << "Num logical CPU cores:  " << port::NumTotalCPUs() << "\n"
            << "Num hyperthreads per core:  " << port::CPUIDNumSMT();
}

void PackNibblesDatapoint(const DatapointPtr<uint8_t>& hash,
                          Datapoint<uint8_t>* packed) {
  const auto hash_size = hash.dimensionality();
  packed->clear();
  packed->set_dimensionality(hash_size);
  packed->mutable_values()->resize((hash_size + 1) / 2, 0);
  PackNibblesDatapoint(hash.values_slice(), packed->mutable_values_slice());
}

void PackNibblesDatapoint(ConstSpan<uint8_t> hash,
                          MutableSpan<uint8_t> packed) {
  DCHECK_EQ(packed.size(), (hash.size() + 1) / 2);
  const auto hash_size = hash.size();
  for (size_t i = 0; i < hash_size / 2; ++i) {
    packed[i] = hash[i * 2 + 1] << 4 | hash[i * 2];
  }
  if (hash_size & 1) {
    packed[hash_size / 2] = hash[hash_size - 1];
  }
}

void UnpackNibblesDatapoint(const DatapointPtr<uint8_t>& packed,
                            Datapoint<uint8_t>* hash) {
  const auto hash_size = packed.dimensionality();
  hash->clear();
  hash->set_dimensionality(hash_size);
  hash->mutable_values()->resize(hash_size, 0);
  UnpackNibblesDatapoint(packed.values_slice(), hash->mutable_values_slice(),
                         hash_size);
}

void UnpackNibblesDatapoint(ConstSpan<uint8_t> packed,
                            MutableSpan<uint8_t> hash,
                            DimensionIndex hash_size) {
  DCHECK_EQ(packed.size(), (hash_size + 1) / 2);
  for (size_t i = 0; i < hash_size / 2; ++i) {
    hash[i * 2] = packed[i] & 0x0f;
    hash[i * 2 + 1] = packed[i] >> 4;
  }
  if (hash_size & 1) {
    hash[hash_size - 1] = packed[hash_size / 2];
  }
}

}  // namespace scann_ops
}  // namespace tensorflow
