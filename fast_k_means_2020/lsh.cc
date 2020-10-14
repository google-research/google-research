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

#include "lsh.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "random_handler.h"

using std::map;
using std::pair;
using std::vector;

namespace fast_k_means {

double LSHDataStructure::SqrDist(const vector<double>& p1,
                                 const vector<double>& p2) {
  double d = 0;
  for (int i = 0; i < p1.size(); i++) {
    d += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  return d;
}

vector<int> LSHDataStructure::Project(const vector<double>& coordinates) {
  vector<int> projections;

  for (int i = 0; i < nb_bins_; i++) {
    int b = projectors_[i].first;
    double c = 0;
    for (int j = 0; j < projectors_[i].second.size(); j++) {
      c += projectors_[i].second[j] * coordinates[j];
    }
    projections.push_back(static_cast<int>((c + b) / r_));
  }

  return projections;
}

void LSHDataStructure::InsertPoint(int id, const vector<double> &coordinates) {
  points_.insert(pair<int, vector<double>>(id, coordinates));

  vector<int> proj = Project(coordinates);
  points_to_bins_.insert(pair<int, vector<int>>(id, proj));

  for (int i = 0; i < nb_bins_; i++) {
    map<int, vector<int>>::iterator it_bin;
    it_bin = bins_collection_[i].find(proj[i]);
    if (it_bin != bins_collection_[i].end()) {
      (it_bin->second).push_back(id);
    } else {
      vector<int> new_bin;
      new_bin.push_back(id);
      bins_collection_[i].insert(pair<int, vector<int>>(proj[i], new_bin));
    }
  }
}

double LSHDataStructure::QueryPoint(const vector<double>& coordinates,
                                    int running_time) {
  // 1. Get the projection: i.e. a list of bins b_1,...,b_nb_bins
  // 2. Consider the elements in the bins b_1,.., b_nb_bins up to a fixed budget
  // 3. Output the closest one.
  vector<int> proj = Project(coordinates);
  int nb_comparisons = 0;
  int id = points_.begin()->first;
  double min_dist = SqrDist(points_.begin()->second, coordinates);

  for (int i = 0; i < nb_bins_; i++) {
    map<int, vector<int>>::iterator it_bin;
    it_bin = bins_collection_[i].find(proj[i]);
    if (it_bin == bins_collection_[i].end()) continue;
    for (int j = 0; j < (it_bin->second).size(); j++) {
      map<int, vector<double>>::iterator p;
      p = points_.find((it_bin->second)[j]);
      double d = SqrDist(coordinates, p->second);
      if (d < min_dist) {
        min_dist = d;
        id = p->first;
        nb_comparisons++;
      }
      // For monotone lsh data structure, please remove the following line. The
      // details about the difference if monotone and non-monotone can be found
      // in the paper.
      if (nb_comparisons > running_time) return min_dist;
    }
  }
  return min_dist;
}

void LSHDataStructure::Print() {
  for (int i = 0; i < nb_bins_; i++) {
    std::cout << "Hash fun #" << i << std::endl;
    for (std::map<int, vector<int>>::iterator it = bins_collection_[i].begin();
         it != bins_collection_[i].end(); ++it) {
      std::cout << "Bin #" << it->first << ":  ";
      for (int j = 0; j < (it->second).size(); j++) {
        std::cout << (it->second)[j] << "   ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }
}

LSHDataStructure::LSHDataStructure(int bucket_size, int nb_bins1,
                                   int dimension) {
  nb_bins_ = nb_bins1;
  r_ = bucket_size;

  std::normal_distribution<double> distrib{0, 1};

  for (int i = 0; i < nb_bins_; i++) {
    int offset = RandomHandler::eng() % r_;
    vector<double> coordinates;
    coordinates.reserve(dimension);
    for (int j = 0; j < dimension; j++) {
      coordinates.push_back(distrib(RandomHandler::eng));
    }
    projectors_.push_back(pair<int, vector<double>>(offset, coordinates));
    map<int, vector<int>> new_map;
    bins_collection_.push_back(new_map);
  }
}

}  //  namespace fast_k_means
