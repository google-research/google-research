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

#ifndef FAIR_SUBMODULAR_MATROID_MOVIES_DATA_H_
#define FAIR_SUBMODULAR_MATROID_MOVIES_DATA_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

// Singleton pattern - there is only one instance of this class that is
// obtained using MoviesData::GetInstance(), and is initialized upon first
// call of GetInstance().
class MoviesData {
 public:
  // Returns the dot product of two movie vectors.
  double GetMovieMovieSimilarity(int movie1, int movie2) const;

  // Returns the dot product of a user vector with a movie vector.
  double GetUserMovieScore(int user, int movie) const;

  // Returns the total number of movies.
  int GetNumberOfMovies() const;

  // Returns the total number of users.
  int GetNumberOfUsers() const;

  // Returns all movie ids.
  const std::vector<int>& GetMovieIds() const;

  // Returns the genre of a movie.
  int GetGenreOfMovie(int movie) const;

  // Returns the full movie id to genre map.
  const absl::flat_hash_map<int, int>& GetMovieIdToGenreIdMap() const;

  // Returns the year band of a movie.
  int GetYearBandOfMovie(int movie) const;

  // Returns the full movie id to year band map.
  const absl::flat_hash_map<int, int>& GetMovieIdToYearBandMap() const;

  // Returns the genre string of a movie by id.
  const std::string& GetGenreStringOfId(int id) const;

  // Given 0-based id of year band, returns std::string like "1911-1920".
  std::string GetYearBandStringOfId(int yb) const;

  // Returns the numeric id of a genre string.
  int GetGenreIdOfString(const std::string& genre) const;

  // Returns lower and upper bounds (as percentages) of each genre, for use in
  // experiments.
  std::vector<std::pair<double, double>> GetMovieGenreBoundPercentages() const;

  // same but for year bands (only upper bounds, for the matroid)
  std::vector<double> GetMovieYearBandBoundPercentages() const;

  // when this is first called, the singleton instance is constructed
  static MoviesData& GetInstance() {
    static auto* const instance = new MoviesData();
    return *instance;
  }

 private:
  // The MovieLens dataset is preprocessed: the sparse user-movie rating matrix
  // given in the dataset is approximated as a product of two low-rank matrices
  // U * V (U holds the user vectors and V holds the movie vectors).

  // Matrices U and V.
  std::vector<std::vector<double>> u_, v_;
  // V * V^T
  std::vector<std::vector<double>> vvt_;
  // Maps movie id to genre id.
  absl::flat_hash_map<int, int> movie_id_to_genre_id_;
  // Maps movie id to year band (as described in paper).
  absl::flat_hash_map<int, int> movie_id_to_year_band_;
  // Vector of movie ids that form the universe.
  std::vector<int> movie_ids_;
  // Maps genre ids to strings like "Drama".
  std::vector<std::string> genre_id_to_string_;
  // Maps genre strings to numerical ids.
  absl::flat_hash_map<std::string, int> genre_string_to_id_;

  // Reads (preprocessed) dataset data.
  MoviesData();

  // Forbid copying.
  MoviesData(const MoviesData&) = delete;
  MoviesData& operator=(const MoviesData&) = delete;
};

#endif  // FAIR_SUBMODULAR_MATROID_MOVIES_DATA_H_
