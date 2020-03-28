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

#include "frequent.h"

#include <map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "sketch.h"
#include "utils.h"

namespace sketch {

Frequent::Frequent(uint heap_size) : heap_size_(heap_size) {}

Frequent::Frequent(const Frequent& other)
    : weight_to_item_(other.weight_to_item_),
      item_to_weight_(other.item_to_weight_),
      heap_size_(other.heap_size_),
      delete_threshold_(other.delete_threshold_) {}

void Frequent::Reset() {
  item_to_weight_.clear();
  weight_to_item_.clear();
  ResetMissing();
}


void Frequent::Add(uint item, float delta) {
  if (auto item_to_weight_it = item_to_weight_.find(item);
      item_to_weight_it != item_to_weight_.end()) {
    const float weight = item_to_weight_it->second->first;
    weight_to_item_.erase(item_to_weight_it->second);
    item_to_weight_[item] = weight_to_item_.emplace(weight + delta, item);
  } else {
    const float adjusted_weight = delta + EstimateMissing(item);
    if (item_to_weight_.size() >= heap_size_) {
      auto smallest_weight_item = weight_to_item_.begin();
      if (adjusted_weight > smallest_weight_item->first) {
        UpdateMissing(smallest_weight_item->second,
                      smallest_weight_item->first);
        item_to_weight_.erase(smallest_weight_item->second);
        weight_to_item_.erase(smallest_weight_item);
        item_to_weight_[item] = weight_to_item_.emplace(adjusted_weight, item);
      } else {
        UpdateMissing(item, adjusted_weight);
      }
    } else {
      item_to_weight_[item] = weight_to_item_.emplace(adjusted_weight, item);
    }
  }
}

float Frequent::Estimate(uint item) const {
  auto iter = item_to_weight_.find(item);
  return iter == item_to_weight_.end() ? EstimateMissing(item)
                                       : iter->second->first;
}

std::vector<uint> Frequent::HeavyHitters(float threshold) const {
  std::vector<uint> items;
  for (auto iter = weight_to_item_.upper_bound(threshold);
       iter != weight_to_item_.cend(); ++iter) {
    items.push_back(iter->second);
  }
  return items;
}

unsigned int Frequent::Size() const {
  return sizeof(Frequent) +
         item_to_weight_.capacity() *
             sizeof(
                 std::pair<uint, decltype(weight_to_item_)::const_iterator>) +
         // The space required for the contents (not pointers) of nodes in the
         // BST (i.e., weight_to_item_).
         weight_to_item_.size() * sizeof(std::pair<float, uint>) +
         // The space required for the pointers of nodes in BST where each node
         // requires three pointers (i.e., parent, left child, and right
         // child).
         weight_to_item_.size() * 3 * sizeof(std::pair<float, uint>&);
}

bool Frequent::Compatible(const Sketch& other_sketch) const {
  const Frequent* other = dynamic_cast<const Frequent*>(&other_sketch);
  if (other == nullptr) return false;
  return CompatibleMissing(*other);
}

void Frequent::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const Frequent& other = static_cast<const Frequent &>(other_sketch);
  // Merge the enrties from other which exist in this first.
  for (const auto& [item, weight_to_item_iter] : item_to_weight_) {
    const float other_weight = other.Estimate(item);
    // One could be lazy here and just say Add(item, other_weight);
    const float weight = weight_to_item_iter->first;
    item_to_weight_[item] =
        weight_to_item_.emplace(weight + other_weight, item);
  }

  // Add remaining items in other but do not exist in this.
  for (const auto& [weight, item] : other.weight_to_item_) {
    if (!item_to_weight_.contains(item)) {
      Add(item, weight);
    }
  }
  MergeMissing(other);
}

}  // namespace sketch
