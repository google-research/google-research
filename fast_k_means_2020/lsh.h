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

// The LSH algorithm we are using is as follows and comes from the paper
// Locality-sensitive hashing scheme based on p-stable distributions by
// Mayur Datar, Nicole Immorlica, Piotr Indyk and Vahab S. Mirrokni.
// - The family of hash functions we are sampling from is defined
//   by (v, r) where v is a random vector where the entries are drawn from
//   N(0,1) and r an integer in [1, bin_size].
// - At creation of the data structure we pick nb_bins such hash functions at
//   random (v1,r1),(v2,r2),...
// - insertion of a point consists in placing it in nb_bins different bins,
//   one bin for each of the hash functions.
//   Point p is placed in bin (i, j) if ((p . vi) + ri)/bin_size = j
// - query of an approximate nearest neighbor of a point p is done in a similar
//   fashion that the insertion:
//   1. find the bins that p would be inserted to
//   2. look at the elements in these bins, for a total of at most running_time
//      elements. Output the one that is the closest.

#ifndef FAST_K_MEANS_2020_LSH_H_
#define FAST_K_MEANS_2020_LSH_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include "random_handler.h"

namespace fast_k_means {

using std::map;
using std::pair;
using std::vector;


class LSHDataStructure {
 public:
  // insert a point with ID id
  void InsertPoint(int id, const vector<double> &coordinates);

  // return the ID (i.e.:the id specified upon insertion)
  // of the elements in the DS that is the ANN
  // Function makes at most running_time comparisons
  double QueryPoint(const vector<double> &coordinates, int running_time);

  // Print elements that have been inserted into the data structure
  // and the bins they belong to.
  void Print();

  // constructor
  LSHDataStructure(int bucket_size, int nb_bins1, int dimension);

 private:
  // Compute L_2 norm between two points
  double SqrDist(const vector<double> &p1, const vector<double> &p2);

  // Project each points onto R n^{1/gamma} times using LSH (this gives the
  // list of bins)
  vector<int> Project(const vector<double> &coordinates);

  // number of hash functions
  int nb_bins_;

  // parameter of LSH; roughly size of the bins
  int r_;

  // maps : id -> coordinates of the inserted points
  map<int, vector<double> > points_;

  // n^{1/gamma} hash functions, each hash function maps to N_+
  vector<pair<int, vector<double> > > projectors_;

  // the non-empty bins of the hash functions (the ones populated by
  // inserted points) and the list
  // of points contained in the bin
  vector<map<int, vector<int> > > bins_collection_;

  // maps each inserted point to the list of bins it belongs to
  map<int, vector<int> > points_to_bins_;
};
}  //  namespace fast_k_means

#endif  // FAST_K_MEANS_2020_LSH_H_
