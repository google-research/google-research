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

#ifndef FAIR_SUBMODULAR_MATROID_MATROID_INTERSECTION_H_
#define FAIR_SUBMODULAR_MATROID_MATROID_INTERSECTION_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "matroid.h"
#include "submodular_function.h"

// Constructs a maximum cardinality set in the intersections of two matroids.
// The solution is constructed in place in both input matroid objects.
void MaxIntersection(Matroid* matroid_a, Matroid* matroid_b,
                     const std::vector<int>& elements);

// `const_elements` are the elements in universe that cannot be touched.
void SubMaxIntersection(Matroid* matroid_a, Matroid* matroid_b,
                        SubmodularFunction* sub_func_f,
                        const absl::flat_hash_set<int>& const_elements,
                        const std::vector<int>& universe);

#endif  // FAIR_SUBMODULAR_MATROID_MATROID_INTERSECTION_H_
