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

#ifndef FAIR_SUBMODULAR_MATROID_CONDITIONED_MATROID_H_
#define FAIR_SUBMODULAR_MATROID_CONDITIONED_MATROID_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "matroid.h"

// This class implements a matroid M', on the *same* universe as `original`
// matroid, such that X is independent in M' iff X u S is independent in the
// original matroid. X can intersect S (indeed any maximal independent set in M'
// will contain the entire S).

// Begins its existence by resetting itself to the empty set.

class ConditionedMatroid : public Matroid {
 public:
  ConditionedMatroid(const Matroid& original, const std::vector<int>& S);

  // Reset to empty set.
  void Reset() override;

  // Return whether adding an element would be feasible.
  bool CanAdd(int element) const override;

  // Return whether add element while removing anothe one would be feasible.
  bool CanSwap(int element, int swap) const override;

  // Add an element. Assumes that the element can be added.
  void Add(int element) override;

  // Removes the element.
  void Remove(int element) override;

  // Checks if a set is feasible.
  bool IsFeasible(const std::vector<int>& elements) const override;

  // Return the current set.
  std::vector<int> GetCurrent() const override;

  // Returns whether an element is in the current set.
  bool InCurrent(int element) const override;

  // Clone the object.
  std::unique_ptr<Matroid> Clone() const override;

 private:
  const absl::flat_hash_set<int> s_;

  // It can intersect with S.
  absl::flat_hash_set<int> current_elements_;

  // It always has S, plus those current_elements that are not from S.
  std::unique_ptr<Matroid> original_;

  // To make Clone() work:
  ConditionedMatroid(const absl::flat_hash_set<int>& s,
                     const absl::flat_hash_set<int>& current_elements,
                     const Matroid& original);
};

#endif  // FAIR_SUBMODULAR_MATROID_CONDITIONED_MATROID_H_
