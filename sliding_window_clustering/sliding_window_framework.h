// Copyright 2022 The Google Research Authors.
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

#ifndef SLIDING_WINDOW_CLUSTERING_SLIDING_WINDOW_FRAMEWORK_H_
#define SLIDING_WINDOW_CLUSTERING_SLIDING_WINDOW_FRAMEWORK_H_

#include "absl/random/random.h"
#include "base.h"

namespace sliding_window {

// This is the class that has to be implemented for a sliding window summary
// algorithm allowing composable sketches.
class SummaryAlg {
 public:
  // window is the window length
  // k is the number of centers
  // lambda is the threshold used by the summary
  // gen is a random be generator
  SummaryAlg(int64_t window, int32_t k, double lambda, absl::BitGen* gen)
      : window_(window),
        k_(k),
        lambda_(lambda),
        gen_(gen),
        is_empty_(true),
        first_element_time_(-1) {}
  virtual ~SummaryAlg() {}

  // Processes one point.
  void process_point(const int64_t time, const std::vector<double>& point);

  // Resets the summary.
  void reset();

  // Returns the time of the first element in the stream,
  int64_t first_element_time() const { return first_element_time_; }

  // Returns whether the stream is empty,
  bool is_empty() const { return is_empty_; }

  // These functions and the ones below must to be implemented by the derived
  // class.
  // Returns the solution and the cost.
  virtual void solution(vector<TimePointPair>* sol, double* value) const = 0;

  // Output the solution resulting from a composion this summary (as summary
  // B -- right summary) with summary_A on the left.
  virtual void solution_left_composition(const SummaryAlg& summary_A,
                                         int64_t window_begin,
                                         vector<TimePointPair>* sol,
                                         double* value) const = 0;

  // Output the result of composing this summary (as summary A) with  and empty
  // summary_B on the right.
  virtual void solution_right_composition_with_empty(int64_t window_begin,
                                                     vector<TimePointPair>* sol,
                                                     double* value) const = 0;

 protected:
  // Actual implementation of prcess_point
  virtual void process_point_impl(const int64_t time,
                                  const std::vector<double>& point) = 0;
  // Actual implementation of reset. Notice that the class must be responsible
  // for keeping updated the number of items stored when the reset function is
  // called.
  virtual void reset_impl() = 0;

 private:
  int32_t window_;
  int32_t k_;
  double lambda_;
  absl::BitGen* gen_;
  bool is_empty_;
  int64_t first_element_time_;
};

// This class implements the procedure to update the pair of summaries
// associated with a threshold lambda.
template <typename SummaryAlg>
class OneThresholdsPairSummaryAlg {
 public:
  // window is the window size.
  // k is the number of clusters
  // lambda is the threshold for the.
  OneThresholdsPairSummaryAlg(int64_t window, int32_t k, double lambda,
                              absl::BitGen* gen)
      : window_(window),
        k_(k),
        lambda_(lambda),
        gen_(gen),
        summary_A_(window_, k_, lambda_, gen_),
        summary_B_(window_, k_, lambda_, gen_) {
    CHECK_GT(lambda, 0);
  }

  virtual ~OneThresholdsPairSummaryAlg() {}

  void process_point(const int64_t time, const std::vector<double>& point) {
    // Add the point to summary B.
    SummaryAlg copy_B = summary_B_;
    summary_B_.process_point(time, point);
    vector<TimePointPair> unused_centers_solution;
    double value;
    summary_B_.solution(&unused_centers_solution, &value);
    // If the value of summary B is < lambda then it is kept in B. Otherwise the
    // new element is the start of a new summary B and summary A is assigned the
    // old value of B.
    if (value >= lambda_) {
      summary_A_ = copy_B;
      summary_B_.reset();
      summary_B_.process_point(time, point);
      first_element_B_ = point;
    }
  }

  const SummaryAlg* get_summary_A() const { return &summary_A_; }

  const SummaryAlg* get_summary_B() const { return &summary_B_; }

 private:
  const int32_t window_;
  const int32_t k_;
  const double lambda_;
  absl::BitGen* gen_;
  SummaryAlg summary_A_;
  SummaryAlg summary_B_;
  std::vector<double> first_element_B_;
};

// This runs the algorithmic framework for our sliding window algorithm.
// The parameters are:
//  delta which is the one in (1+delta) used for the grid of costs
//  window which is the window size
//  begin grid which is used for the begin of the grid
//  end grid which is the end of the grid.
template <typename SummaryAlg>
class FrameworkAlg {
 public:
  FrameworkAlg(int64_t window, int32_t k, double delta, double begin_grid,
               double end_grid, absl::BitGen* gen)
      : window_(window),
        k_(k),
        delta_(delta),
        begin_grid_(begin_grid),
        end_grid_(end_grid),
        gen_(gen),
        window_handler_(window) {
    CHECK_GT(delta, 0);
    CHECK_LT(begin_grid, end_grid);
    CHECK_GT(window_, 1);

    // Initializes the summary thresholds.
    double lambda = begin_grid_;
    while (lambda <= end_grid_ * (1.0 + delta)) {
      threshold_algs_.push_back(
          OneThresholdsPairSummaryAlg<SummaryAlg>(window_, k_, lambda, gen_));
      lambda *= (1 + delta);
    }
  }
  // Disllow copy and assign.
  FrameworkAlg(FrameworkAlg const&) = delete;
  void operator=(FrameworkAlg const& x) = delete;

  // Processes a point.
  void process_point(const int64_t time, const std::vector<double>& point) {
    window_handler_.next();
    CHECK_EQ(time, window_handler_.curr_time());
    for (auto& threshold_alg : threshold_algs_) {
      threshold_alg.process_point(time, point);
    }
  }

  // Outputs the solution and its cost.
  void solution(vector<TimePointPair>* solution, double* value) {
    // Contrary to the simplified version in the main body of the paper, the
    // algorithm considers any valid solution (i.e. computer over the entire
    // window) and outputs the one with the lowest estimate of cost, this only
    // improves the results.
    // Check if there is B_lambda that is = active_window.
    double min_cost = std::numeric_limits<double>::max();
    vector<TimePointPair> best_solution;

    for (const auto& threshold_alg : threshold_algs_) {
      if (!threshold_alg.get_summary_B()->is_empty() &&
          threshold_alg.get_summary_B()->first_element_time() ==
              window_handler_.begin_window()) {
        threshold_alg.get_summary_B()->solution(solution, value);
        if (*value < min_cost) {
          min_cost = *value;
          best_solution = *solution;
        }
      } else if (!threshold_alg.get_summary_B()->is_empty() &&
                 threshold_alg.get_summary_B()->first_element_time() <
                     window_handler_.begin_window()) {
        threshold_alg.get_summary_B()->solution_right_composition_with_empty(
            window_handler_.begin_window(), solution, value);
        if (*value < min_cost) {
          min_cost = *value;
          best_solution = *solution;
        }
      }
    }

    // This is a backup in case the algorithmic guarantess of the summaries
    // fails (or if the lower or upper bound are wrong) which is possible in
    // practice. It will output the solution of the sketch (subset of the window
    // that has longest possible history).
    vector<TimePointPair> oldest_sketch_solution;
    double oldest_sketch_cost;
    int64_t oldest_begin_first = std::numeric_limits<int64_t>::max();

    for (const auto& threshold_alg : threshold_algs_) {
      // A is not empty and it is not a subset of W for the smallest lambda.
      if (!threshold_alg.get_summary_A()->is_empty() &&
          threshold_alg.get_summary_A()->first_element_time() <
              window_handler_.begin_window()) {
        // Either W is a strict subset of B.
        if (threshold_alg.get_summary_B()->first_element_time() <
            window_handler_.begin_window()) {
          threshold_alg.get_summary_B()->solution_right_composition_with_empty(
              window_handler_.begin_window(), solution, value);
          if (*value < min_cost) {
            min_cost = *value;
            best_solution = *solution;
          }
        } else {
          threshold_alg.get_summary_B()->solution_left_composition(
              *threshold_alg.get_summary_A(), window_handler_.begin_window(),
              solution, value);
          if (*value < min_cost) {
            min_cost = *value;
            best_solution = *solution;
          }
        }
      } else {  // backup for the case with no guarantees.
        if (!threshold_alg.get_summary_A()->is_empty() &&
            threshold_alg.get_summary_A()->first_element_time() <
                oldest_begin_first) {
          oldest_begin_first =
              threshold_alg.get_summary_A()->first_element_time();
          threshold_alg.get_summary_B()->solution_left_composition(
              *threshold_alg.get_summary_A(), 0, &oldest_sketch_solution,
              &oldest_sketch_cost);
        }
      }
    }
    *solution = best_solution;
    *value = min_cost;
    if (min_cost < std::numeric_limits<double>::max()) {
      return;
    }
    // This point is reached in case of failure of the sketches to give approx.
    // guarantees or if the upper-lower bounds of the cost are wrong. In this
    // case
    *solution = oldest_sketch_solution;
    *value = oldest_sketch_cost;
    CHECK_LT(oldest_begin_first, std::numeric_limits<int64_t>::max());
  }

 private:
  const int64_t window_;
  const int32_t k_;
  const double delta_;
  const double begin_grid_;
  const double end_grid_;
  absl::BitGen* gen_;
  SlidingWindowHandler window_handler_;
  vector<OneThresholdsPairSummaryAlg<SummaryAlg>> threshold_algs_;
};

}  //  namespace sliding_window

#endif  // SLIDING_WINDOW_CLUSTERING_MEYERSON_ALGORITHM_H_
