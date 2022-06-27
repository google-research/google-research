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

#ifndef INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_HELPERS_H_
#define INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_HELPERS_H_

#include <set>

#include "glog/logging.h"
#include "absl/container/flat_hash_map.h"

// Creates a map from label classes to IDs. The IDs are assigned in alphabetical
// order, starting with 0. T must be a proto message that has a repeated field
// called "entry", and "entry" must be a message with a field called
// "target_class_string".
template <typename T>
absl::flat_hash_map<std::string, int> CreatePredictionToIdMap(
    const T& label_class_map) {
  // Create set of possible prediction class strings.
  std::set<std::string> possible_target_classes;
  for (const auto& mapping : label_class_map.entry()) {
    possible_target_classes.insert(mapping.target_class_string());
  }

  for (const auto& implicit_class : label_class_map.implicit_target_classes()) {
    possible_target_classes.erase(implicit_class);
  }

  // std::set elements are iterated through in sorted order, with the default
  // sorting for strings being based on alphabetical order.
  absl::flat_hash_map<std::string, int> predictor_to_id_map;
  int id = 0;
  for (const auto& target_class : possible_target_classes) {
    predictor_to_id_map.emplace(target_class, id);
    id++;
  }

  return predictor_to_id_map;
}

#endif  // INTENT_RECOGNITION_PROCESSING_CLASS_MAPPINGS_PROVIDER_HELPERS_H_
