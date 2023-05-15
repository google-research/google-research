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

#include "movies_data.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "utilities.h"

namespace {

std::vector<std::vector<double>> ReadMatrixFromFile(
    const std::string& filename) {
  std::ifstream f(filename);
  if (!f) {
    Fail("movies file does not exist. run prepare_movies.py first?");
  }
  int rows, cols;
  f >> rows >> cols;
  std::vector<std::vector<double>> res;
  for (int i = 0; i < rows; ++i) {
    res.emplace_back(cols);  // add new row
    for (int j = 0; j < cols; ++j) {
      f >> res[i][j];
    }
  }
  return res;
}

std::vector<std::vector<double>> Transpose(
    const std::vector<std::vector<double>>& A) {
  std::vector<std::vector<double>> res(A[0].size(),
                                       std::vector<double>(A.size()));
  for (int i = 0; i < A.size(); ++i) {
    for (int j = 0; j < A[0].size(); ++j) {
      res[j][i] = A[i][j];
    }
  }
  return res;
}

std::vector<std::vector<double>> ComputeMovieMovieSimilarityMatrix(
    const std::vector<std::vector<double>>& V) {
  std::vector<std::vector<double>> res(V.size(), std::vector<double>(V.size()));
  for (int i = 0; i < V.size(); ++i) {
    for (int j = 0; j < V.size(); ++j) {
      res[i][j] = 0.0;
      for (int k = 0; k < V[0].size(); ++k) {
        res[i][j] += V[i][k] * V[j][k];
      }
    }
  }
  return res;
}

std::string DetermineMainGenre(std::vector<std::string> genres) {
  assert(!genres.empty());
  if (genres.size() == 1) return genres[0];
  // if 'Drama' is one of the genres but not the only one, discard it
  genres.erase(std::remove(genres.begin(), genres.end(), "Drama"),
               genres.end());
  // pick randomly
  return genres[RandomHandler::generator_() % genres.size()];
}

}  // namespace

MoviesData::MoviesData() {
  const std::string path = "";
  std::cerr << "Reading movie data...\n";
  u_ = ReadMatrixFromFile(path + "U.txt");
  v_ = Transpose(ReadMatrixFromFile(path + "VT.txt"));
  assert(u_[0].size() == v_[0].size());
  vvt_ = ComputeMovieMovieSimilarityMatrix(v_);

  std::ifstream dat(path + "movies.dat");
  if (!dat) {
    Fail("movies file does not exist");
  }
  std::string line;
  absl::flat_hash_map<int, std::string> idToGenre;
  while (std::getline(dat, line)) {
    if (line.size() > 2) {
      // a line looks like this:
      // 1::Toy Story (1995)::Animation|Children's|Comedy
      int pos = 0;
      while (line[pos] != ':') ++pos;
      int id = std::stoi(line.substr(0, pos)) - 1;  // 0-based index

      pos = line.size() - 1;
      while (line[pos] != '(') --pos;
      int year = std::stoi(line.substr(pos + 1, 4));
      assert(1919 <= year && year <= 2000);
      movie_id_to_year_band_[id] = (year - 1911) / 10;

      int pos2 = line.size();
      pos = pos2 - 1;
      std::vector<std::string> genres;
      while (true) {
        while (line[pos] != ':' && line[pos] != '|') --pos;
        std::string genre = line.substr(pos + 1, pos2 - pos - 1);
        genres.push_back(genre);
        if (line[pos] == ':') break;
        pos2 = pos;
        --pos;
      }
      std::string mainGenre = DetermineMainGenre(genres);
      idToGenre[id] = mainGenre;
      genre_id_to_string_.push_back(mainGenre);
      // genre_id_to_string is now an unsorted list with possible duplicates.
    }
  }

  std::sort(genre_id_to_string_.begin(), genre_id_to_string_.end());
  genre_id_to_string_.erase(
      std::unique(genre_id_to_string_.begin(), genre_id_to_string_.end()),
      genre_id_to_string_.end());
  // genre_id_to_string_ is now sorted and without duplicates.
  for (int g = 0; g < genre_id_to_string_.size(); ++g) {
    genre_string_to_id_[genre_id_to_string_[g]] = g;
  }

  for (std::pair<int, std::string> p : idToGenre) {
    int id = p.first;
    std::string genre = p.second;
    movie_id_to_genre_id_[id] = genre_string_to_id_[genre];
    movie_ids_.push_back(id);
  }

  std::cerr << "Movie data ready\n";
}

int MoviesData::GetGenreOfMovie(int movie) const {
  return movie_id_to_genre_id_.at(movie);
}

int MoviesData::GetYearBandOfMovie(int movie) const {
  return movie_id_to_year_band_.at(movie);
}

const absl::flat_hash_map<int, int>& MoviesData::GetMovieIdToGenreIdMap()
    const {
  return movie_id_to_genre_id_;
}

const absl::flat_hash_map<int, int>& MoviesData::GetMovieIdToYearBandMap()
    const {
  return movie_id_to_year_band_;
}

double MoviesData::GetMovieMovieSimilarity(int movie1, int movie2) const {
  return vvt_[movie1][movie2];
}

double MoviesData::GetUserMovieScore(int user, int movie) const {
  double res = 0.;
  for (int k = 0; k < v_[0].size(); ++k) {
    res += u_[user][k] * v_[movie][k];
  }
  return res;
}

int MoviesData::GetNumberOfMovies() const { return v_.size(); }

int MoviesData::GetNumberOfUsers() const { return u_.size(); }

const std::vector<int>& MoviesData::GetMovieIds() const { return movie_ids_; }

const std::string& MoviesData::GetGenreStringOfId(int id) const {
  assert(0 <= id && id < genre_id_to_string_.size());
  return genre_id_to_string_[id];
}

std::string MoviesData::GetYearBandStringOfId(int yb) const {
  assert(0 <= yb && yb < 9);
  return absl::StrCat(1911 + yb * 10, "-", 1920 + yb * 10);
}

int MoviesData::GetGenreIdOfString(const std::string& genre) const {
  assert(genre_string_to_id_.contains(genre));
  return genre_string_to_id_.at(genre);
}

std::vector<std::pair<double, double>>
MoviesData::GetMovieGenreBoundPercentages() const {
  const int no_genres = genre_id_to_string_.size();
  std::vector<int> count_movies_of_genre(no_genres, 0);
  for (int id : movie_ids_) {
    count_movies_of_genre[movie_id_to_genre_id_.at(id)]++;
  }

  // 0 Action 256
  // 1 Adventure 132
  // 2 Animation 43
  // 3 Children's 108
  // 4 Comedy 920
  // 5 Crime 130
  // 6 Documentary 121
  // 7 Drama 843
  // 8 Fantasy 22
  // 9 Film-Noir 22
  // 10 Horror 255
  // 11 Musical 69
  // 12 Mystery 65
  // 13 Romance 313
  // 14 Sci-Fi 130
  // 15 Thriller 312
  // 16 War 92
  // 17 Western 50
  constexpr double lower_coeff = 0.8, upper_coeff = 1.4;
  const int no_movies = movie_ids_.size();

  std::vector<std::pair<double, double>> percentages;
  for (int g = 0; g < no_genres; ++g) {
    double lower_bound = lower_coeff * count_movies_of_genre[g] / no_movies;
    double upper_bound = upper_coeff * count_movies_of_genre[g] / no_movies;
    percentages.emplace_back(lower_bound, upper_bound);
  }
  return percentages;
}

std::vector<double> MoviesData::GetMovieYearBandBoundPercentages() const {
  const int noYearBands = 9;
  std::vector<int> countMoviesOfYearBand(noYearBands, 0);
  for (int id : movie_ids_) {
    countMoviesOfYearBand[movie_id_to_year_band_.at(id)]++;
  }

  const int noMovies = movie_ids_.size();

  std::vector<double> percentages;
  for (int yb = 0; yb < noYearBands; ++yb) {
    const double upper_coeff = 1.2;  // (yb == noYearBands-1) ? 1.03 : 1.2;
    double upper_bound = upper_coeff * countMoviesOfYearBand[yb] / noMovies;
    percentages.emplace_back(upper_bound);
  }
  return percentages;
}
