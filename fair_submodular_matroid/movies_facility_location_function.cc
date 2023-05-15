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

#include "movies_facility_location_function.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "movies_data.h"

MoviesFacilityLocationFunction::MoviesFacilityLocationFunction()
    : max_sim_(MoviesData::GetInstance().GetNumberOfMovies(), {0.}) {}

void MoviesFacilityLocationFunction::Reset() {
  for (auto& ref : max_sim_) {
    ref = {0.};
  }
}

double MoviesFacilityLocationFunction::Delta(int movie) {
  double res = 0.;
  const int noMovies = MoviesData::GetInstance().GetNumberOfMovies();
  for (int i = 0; i < noMovies; ++i) {
    res += std::max(
        0.0, MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i) -
                 *max_sim_[i].begin());
  }
  return res;
}

void MoviesFacilityLocationFunction::Remove(int movie) {
  const int no_movies = MoviesData::GetInstance().GetNumberOfMovies();
  for (int i = 0; i < no_movies; ++i) {
    auto it = max_sim_[i].find(
        MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i));
    assert(it != max_sim_[i].end());
    max_sim_[i].erase(it);
  }
}

double MoviesFacilityLocationFunction::RemovalDelta(int movie) {
  double val = 0.;
  const int noMovies = MoviesData::GetInstance().GetNumberOfMovies();
  for (int i = 0; i < noMovies; ++i) {
    const double eval =
        MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i);
    auto it = max_sim_[i].begin();
    if (*it == eval) {
      // Movie has the maximum currently, so we look at the second-best.
      ++it;
      val += eval - *it;
    }  // else: movie is not the maximum, so removing it won't change things.
  }
  return val;
}

// Not necessary, but overloaded for efficiency
double MoviesFacilityLocationFunction::RemoveAndIncreaseOracleCall(int movie) {
  ++oracle_calls_;
  double val = 0.;
  const int no_movies = MoviesData::GetInstance().GetNumberOfMovies();
  for (int i = 0; i < no_movies; ++i) {
    const double before = *max_sim_[i].begin();
    auto it = max_sim_[i].find(
        MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i));
    assert(it != max_sim_[i].end());
    max_sim_[i].erase(it);
    const double after = *max_sim_[i].begin();
    val += before - after;
  }
  return val;
}

void MoviesFacilityLocationFunction::Add(int movie) {
  const int no_movies = MoviesData::GetInstance().GetNumberOfMovies();
  for (int i = 0; i < no_movies; ++i) {
    max_sim_[i].insert(
        MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i));
  }
}

double MoviesFacilityLocationFunction::Objective(
    const std::vector<int>& elements) const {
  // stateless implementation
  const int noMovies = MoviesData::GetInstance().GetNumberOfMovies();
  double res = 0.;
  for (int i = 0; i < noMovies; ++i) {
    double max_sim_i = 0.;
    for (int movie : elements) {
      max_sim_i =
          std::max(max_sim_i,
                   MoviesData::GetInstance().GetMovieMovieSimilarity(movie, i));
    }
    res += max_sim_i;
  }
  return res;
}

const std::vector<int>& MoviesFacilityLocationFunction::GetUniverse() const {
  return MoviesData::GetInstance().GetMovieIds();
}

std::string MoviesFacilityLocationFunction::GetName() const {
  return "MovieLens (facility-location objective)";
}

std::unique_ptr<SubmodularFunction> MoviesFacilityLocationFunction::Clone()
    const {
  return std::make_unique<MoviesFacilityLocationFunction>(*this);
}
