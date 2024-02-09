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

#ifndef FAIR_SUBMODULAR_MATROID_MATROID_H_
#define FAIR_SUBMODULAR_MATROID_MATROID_H_

#include <memory>
#include <vector>

class Matroid {
 public:
  virtual ~Matroid() = default;

  // Reset to empty set.
  virtual void Reset() = 0;

  // Return whether adding an element would be feasible.
  virtual bool CanAdd(int element) const = 0;

  // Return whether add element while removing anothe one would be feasible.
  virtual bool CanSwap(int element, int swap) const = 0;

  // Returns all possible swaps for a given new element.
  virtual std::vector<int> GetAllSwaps(int elements) const;

  // Add an element. Assumes that the element can be added.
  virtual void Add(int element) = 0;

  // Swap an element for one already in the solution.
  virtual void Swap(int element, int swap);

  // Removes the element.
  virtual void Remove(int element) = 0;

  // Checks if a set is feasible.
  virtual bool IsFeasible(const std::vector<int>& elements) const = 0;

  // Checks whether the current set is feasible.
  virtual bool CurrentIsFeasible() const;

  // Return the current set.
  virtual std::vector<int> GetCurrent() const = 0;

  // Returns whether an element is in the current set.
  virtual bool InCurrent(int element) const = 0;

  // Clone the object (see e.g. GraphUtility for an example).
  virtual std::unique_ptr<Matroid> Clone() const = 0;
};

#endif  // FAIR_SUBMODULAR_MATROID_MATROID_H_
