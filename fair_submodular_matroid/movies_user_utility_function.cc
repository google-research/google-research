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

#include "movies_user_utility_function.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "movies_data.h"

MoviesUserUtilityFunction::MoviesUserUtilityFunction(int user) : user_(user) {
  assert(user_ >= 0 && user_ < MoviesData::GetInstance().GetNumberOfUsers());
}

void MoviesUserUtilityFunction::Reset() { present_elements_.clear(); }

double MoviesUserUtilityFunction::Delta(int movie) {
  assert(movie >= 0 && movie < MoviesData::GetInstance().GetNumberOfMovies());
  if (present_elements_.count(movie)) {
    return 0.;
  }
  return std::max(0.,
                  MoviesData::GetInstance().GetUserMovieScore(user_, movie));
}

double MoviesUserUtilityFunction::RemovalDelta(int movie) {
  assert(movie >= 0 && movie < MoviesData::GetInstance().GetNumberOfMovies());
  assert(present_elements_.count(movie));
  return std::max(0.0,
                  MoviesData::GetInstance().GetUserMovieScore(user_, movie));
}

void MoviesUserUtilityFunction::Remove(int movie) {
  assert(movie >= 0 && movie < MoviesData::GetInstance().GetNumberOfMovies());
  assert(present_elements_.erase(movie) == 1);
}

void MoviesUserUtilityFunction::Add(int movie) {
  present_elements_.insert(movie);
}

double MoviesUserUtilityFunction::Objective(
    const std::vector<int>& elements) const {
  // stateless implementation
  double res = 0.0;
  for (int movie : elements) {
    assert(movie >= 0 && movie < MoviesData::GetInstance().GetNumberOfMovies());
    res += std::max(0.0,
                    MoviesData::GetInstance().GetUserMovieScore(user_, movie));
  }
  return res;
}

const std::vector<int>& MoviesUserUtilityFunction::GetUniverse() const {
  return MoviesData::GetInstance().GetMovieIds();
}

std::string MoviesUserUtilityFunction::GetName() const {
  return absl::StrCat("MovieLens (utility for user_ ", user_, ")");
}

std::unique_ptr<SubmodularFunction> MoviesUserUtilityFunction::Clone() const {
  return std::make_unique<MoviesUserUtilityFunction>(*this);
}
