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

#include "movies_mixed_utility_function.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "movies_data.h"

MoviesMixedUtilityFunction::MoviesMixedUtilityFunction(int user, double alpha)
    : mf_(), mu_(user), alpha_(alpha) {}

void MoviesMixedUtilityFunction::Reset() {
  mf_.Reset();
  mu_.Reset();
}

double MoviesMixedUtilityFunction::Delta(int movie) {
  return alpha_ * mf_.Delta(movie) + (1 - alpha_) * mu_.Delta(movie);
}

double MoviesMixedUtilityFunction::RemovalDelta(int movie) {
  return alpha_ * mf_.RemovalDelta(movie) +
         (1 - alpha_) * mu_.RemovalDelta(movie);
}

void MoviesMixedUtilityFunction::Add(int movie) {
  mu_.Add(movie);
  mf_.Add(movie);
}

void MoviesMixedUtilityFunction::Remove(int movie) {
  mu_.Remove(movie);
  mf_.Remove(movie);
}

// Not necessary, but overloaded for efficiency
double MoviesMixedUtilityFunction::RemoveAndIncreaseOracleCall(int movie) {
  --oracle_calls_;  // Since the below line will incur two oracle calls.
  return alpha_ * mf_.RemoveAndIncreaseOracleCall(movie) +
         (1 - alpha_) * mu_.RemoveAndIncreaseOracleCall(movie);
}

double MoviesMixedUtilityFunction::Objective(
    const std::vector<int>& movies) const {
  return alpha_ * mf_.Objective(movies) + (1 - alpha_) * mu_.Objective(movies);
}

const std::vector<int>& MoviesMixedUtilityFunction::GetUniverse() const {
  return MoviesData::GetInstance().GetMovieIds();
}

std::string MoviesMixedUtilityFunction::GetName() const {
  return absl::StrCat("mix of: ", alpha_, " of ", mf_.GetName(), " and ",
                      1 - alpha_, " of ", mu_.GetName());
}

std::unique_ptr<SubmodularFunction> MoviesMixedUtilityFunction::Clone() const {
  return std::make_unique<MoviesMixedUtilityFunction>(*this);
}
