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

#ifndef SKETCHING_SKETCH_H_
#define SKETCHING_SKETCH_H_

// Interface for a sketch. Each class will implement these methods,
// but possibly others too.

#include <vector>

#include "utils.h"

namespace sketch {

class Sketch {
 public:
  Sketch() {}
  virtual ~Sketch() {}

  // Resets the values recorded in the sketch (but not any hashes etc).
  virtual void Reset() = 0;

  // Adds delta to the value of an item. For count, use delta = 1
  virtual void Add(uint item, float delta) = 0;

  // Do any cleanup needed to prepare for an estimation, heavy hitters etc.
  virtual void ReadyToEstimate() {}

  // Returns the estimated value of an item
  virtual float Estimate(uint item) const = 0;

  // Returns the list of elements with estimated counts above threshold.
  virtual std::vector<uint> HeavyHitters(float threshold) const = 0;

  // Returns the amount of memory (in bytes) used by the sketch.
  virtual uint Size() const = 0;

  // Check if the other sketch is compatible with this - uses same hashes etc.
  virtual bool Compatible(const Sketch& other_sketch) const = 0;

  // Merges the other sketch with this. Does nothing if not compatible.
  // Note that it is possible to merge incompatible sketches too,
  // but that is very expensive so is not implemented.
  virtual void Merge(const Sketch& other_sketch) = 0;
};

}  // namespace sketch

#endif  // SKETCHING_SKETCH_H_
