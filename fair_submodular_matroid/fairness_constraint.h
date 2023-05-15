// Copyright 2023 The Authors.
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

#ifndef FAIR_SUBMODULAR_MATROID_FAIRNESS_CONSTRAINT_H_
#define FAIR_SUBMODULAR_MATROID_FAIRNESS_CONSTRAINT_H_

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "matroid.h"

class FairnessConstraint {
 public:
  FairnessConstraint(const absl::flat_hash_map<int, int>& colors_map,
                     const std::vector<std::pair<int, int>>& bounds);

  ~FairnessConstraint() = default;

  // Reset to empty set.
  void Reset();

  // Return whether adding an element would be feasible.
  bool CanAdd(int element) const;

  // Add an element. Assumes that the element can be added.
  void Add(int element);

  // Return whether removing an element would be feasible.
  bool CanRemove(int element) const;

  // Removes the element. Assumes that the element can be removed.
  void Remove(int element);

  // Checks if a set is feasible.
  bool IsFeasible(std::vector<int> elements);

  // Returns color of an element.
  int GetColor(int element) const;

  // Returns the number of colors.
  int GetColorNum() const;

  // Return a partition matroid based on color classes and lower bonuds.
  std::unique_ptr<Matroid> LowerBoundsToMatroid() const;

  // Return a partition matroid based on color classes and upper bonuds.
  std::unique_ptr<Matroid> UpperBoundsToMatroid() const;

  // Clone the object.
  std::unique_ptr<FairnessConstraint> Clone() const;

  // Returns the bounds.
  std::vector<std::pair<int, int>> GetBounds() const;

 private:
  // Map universe elements to groups
  absl::flat_hash_map<int, int> colors_map_;

  // Color bounds (lower, upper) pairs
  std::vector<std::pair<int, int>> bounds_;
  int ncolors_;

  // Current number of elements per color
  std::vector<int> current_colorcounts_;

  // Current set
  std::set<int> current_set_;
};

#endif  // FAIR_SUBMODULAR_MATROID_FAIRNESS_CONSTRAINT_H_
