// Copyright 2021 The Google Research Authors.
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

// Implementation of basic functions and data structures needed for the sliding
// window algorithm
#ifndef SLIDING_WINDOW_CLUSTERING_BASE_H_
#define SLIDING_WINDOW_CLUSTERING_BASE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "absl/random/random.h"

#define CHECK_EQ(a, b) (assert((a) == (b)))
#define CHECK_GT(a, b) (assert((a) > (b)))
#define CHECK_GE(a, b) (assert((a) >= (b)))
#define CHECK_LT(a, b) (assert((a) < (b)))
#define CHECK_LE(a, b) (assert((a) <= (b)))
#define CHECK(a) (assert((a)))

namespace sliding_window {

using std::pair;
using std::vector;
using std::string;

// Extern global variable storing the total count of all calls to a distance
// function computation.
extern int64_t CALLS_DIST_FN;
// Extern global  count of the *current* number of points stored in any summary.
extern int64_t ITEMS_STORED;

// Pair of arrival time and a point represented as a vector of doubles.
// The arrivial time acts as an id of the point.
using TimePointPair = pair<int64_t, vector<double>>;

// Class representing a stream of points.
class Stream {
 public:
  Stream() {}
  virtual ~Stream() {}

  virtual TimePointPair next_point() = 0;
  virtual bool has_next() = 0;
};

// Computes the L2 Distance between two vectors. The function updates the
// CALLS_DIST_FN counter.
double l2_distance(const vector<double>& a, const vector<double>& b);

// Computes the  L2 Distance between two vectors in TimePointPair format. The
// function updates the CALLS_DIST_FN counter.
double l2_distance(const TimePointPair& a, const TimePointPair& b);

// Implementation of KMeans++ for weighted point sets. The input consists of a
// fector of points and weights. The output is the cost of the centers on the
// weighted instance and the set of centers (as positions in the input vector).
void k_means_plus_plus(
    const std::vector<std::pair<TimePointPair, double>>& instance,
    const int32_t k, absl::BitGen* gen, std::vector<int32_t>* centers,
    double* cost);

// Outputs the k-means cost of a clustering solution.
double cost_solution(const std::vector<TimePointPair>& instance,
                     const std::vector<TimePointPair>& centers);

// Returns the optimal assignment of points to the nearest center.
void cluster_assignment(const std::vector<TimePointPair>& instance,
                        const std::vector<TimePointPair>& centers,
                        std::vector<int32_t>* assignment);

// Class that handles the time stamps of a sliding window. The timestamps start
// from 0.
class SlidingWindowHandler {
 public:
  SlidingWindowHandler(const int window_size)
      : curr_time_(-1), begin_window_(0), window_size_(window_size) {}

  // Increase time counter.
  void next() {
    curr_time_++;
    if (curr_time_ >= window_size_) {
      begin_window_++;
    }
    CHECK_LE(curr_time_ - begin_window_ + 1, window_size_);
  }

  // Current time.
  inline int64_t curr_time() const { return curr_time_; }
  // Beginning of the window.
  inline int64_t begin_window() const { return begin_window_; }

 private:
  int64_t curr_time_;
  int64_t begin_window_;
  const int32_t window_size_;
};


}  //  namespace sliding_window

#endif  // SLIDING_WINDOW_CLUSTERING_BASE_H_
