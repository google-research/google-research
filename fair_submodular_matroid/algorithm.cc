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

#include "algorithm.h"

#include <cassert>
#include <memory>

#include "fairness_constraint.h"
#include "matroid.h"
#include "submodular_function.h"

void Algorithm::Init(const SubmodularFunction& sub_func_f,
                     const FairnessConstraint& fairness,
                     const Matroid& matroid) {
  sub_func_f_ = sub_func_f.Clone();
  sub_func_f_->Reset();
  fairness_ = fairness.Clone();
  fairness_->Reset();
  matroid_ = matroid.Clone();
  matroid_->Reset();
}

int Algorithm::GetNumberOfPasses() const {
  // default is one-pass
  return 1;
}

void Algorithm::BeginNextPass() {
  // multi-pass algorithms should overload this, so if we ever reach here,
  // this means that the algorithm is single-pass (so this should not be called)
  assert(false);
}
