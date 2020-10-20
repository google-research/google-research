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

#ifndef SLIDING_WINDOW_CLUSTERING_MEYERSON_ALGORITHM_H_
#define SLIDING_WINDOW_CLUSTERING_MEYERSON_ALGORITHM_H_

#include <deque>
#include <limits>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "base.h"
#include "sliding_window_framework.h"

namespace sliding_window {

// This class implements an estimate of the total weight inserted after a
// certain time in a sliding window. The class assumes that the times of
// insertion are inserted in increasing order. This can be used to keep track of
// the number of items inserted in a bucket after a certain tiem or to keep
// track of the total cost of the bucket.
class ApproxTimeCountKeeper {
 public:
  // Epsilon is the approximation error allowed (i.e., we allow 1+epsilon).
  explicit ApproxTimeCountKeeper(double epsilon) : epsilon_(epsilon) {
    CHECK_GT(epsilon_, 0);
  }
  ~ApproxTimeCountKeeper() {}

  // Increase the count by 'how_much' at time 'time'.
  void increase_total(const int64_t time, const double how_much) {
    for (auto& time_count : time_counts_) {
      CHECK_LT(time_count.first, time);
      time_count.second += how_much;
    }

    time_counts_.push_back(std::make_pair(time, how_much));

    auto it = time_counts_.begin();
    // Compact indices if they are not too different.
    while (true) {
      auto it2 = it;
      // Move twice ahead if possible.
      if (it2 == time_counts_.end()) {
        break;
      }
      it2++;
      if (it2 == time_counts_.end()) {
        break;
      }
      auto it_middle = it2;
      it2++;
      if (it2 == time_counts_.end()) {
        break;
      }
      // Check if need to remove middle
      if (it->second <= (1.0 + epsilon_) * it2->second) {
        it_middle = time_counts_.erase(it_middle);
        it = it_middle;
      } else {
        ++it;
      }
    }
  }

  // Returns a 1+epsilon estimate of the total weight added to the counter after
  // begin_time.
  double total_after_time(const int64_t begin_time) const {
    double count = time_counts_.front().second;
    for (const auto& time_count : time_counts_) {
      if (time_count.first <= begin_time) {
        count = time_count.second;
      }
    }
    return count;
  }

  // Returns the count at 0.
  double count_at_0() const { return time_counts_.front().second; }

 private:
  std::list<std::pair<int64_t, double>> time_counts_;
  // Approximation error.
  double epsilon_;
};

// This class implements the well-known meyerson sketch for k-means with all the
// bookkeping needed for the sliding window algorithm. This class implements a
// single sketch.
class MeyersonSketch {
 public:
  // max_num_centers is the maximum number of centers allowed in the sketch.
  // denominator_prob is the denominator in the probability of selecting a point
  // as center.
  // epsilon_multiplicities is the epsilon factor used in the estimate of the
  // multiplicites of the centers. k is the target number of centers. gen is a
  // random source.
  MeyersonSketch(const double max_num_centers, const double denominator_prob,
                 const double epsilon_multiplicities, const int32_t k,
                 absl::BitGen* gen)
      : max_num_centers_(max_num_centers),
        denominator_prob_(denominator_prob),
        epsilon_multiplicities_(epsilon_multiplicities),
        k_(k),
        gen_(gen),
        failed_sketch_(false) {}
  // Notice how the number of items stored is updated by the destructor.
  virtual ~MeyersonSketch() { ITEMS_STORED -= centers_.size(); }

  // Copy operator. Handles the number of items stored.
  MeyersonSketch(MeyersonSketch const& other) {
    max_num_centers_ = other.max_num_centers_;
    denominator_prob_ = other.denominator_prob_;
    epsilon_multiplicities_ = other.epsilon_multiplicities_;
    k_ = other.k_;
    gen_ = other.gen_;
    failed_sketch_ = other.failed_sketch_;
    centers_ = other.centers_;
    ITEMS_STORED += centers_.size();
    multiplicities_ = other.multiplicities_;
    costs_sum_dist_ = other.costs_sum_dist_;
    costs_sum_sq_dist_ = other.costs_sum_sq_dist_;
  }

  MeyersonSketch& operator=(MeyersonSketch const& other) {
    max_num_centers_ = other.max_num_centers_;
    denominator_prob_ = other.denominator_prob_;
    epsilon_multiplicities_ = other.epsilon_multiplicities_;
    k_ = other.k_;
    gen_ = other.gen_;
    failed_sketch_ = other.failed_sketch_;
    // The previous number of centers is freed.
    ITEMS_STORED -= centers_.size();
    centers_ = other.centers_;
    ITEMS_STORED += centers_.size();
    multiplicities_ = other.multiplicities_;
    costs_sum_dist_ = other.costs_sum_dist_;
    costs_sum_sq_dist_ = other.costs_sum_sq_dist_;
    return *this;
  }

  // Add a point
  void add_point(const TimePointPair& point) {
    if (failed_sketch_) {
      return;
    }
    if (centers_.empty()) {
      create_center(point);
      return;
    }
    double min_distance = std::numeric_limits<double>::max();
    int64_t best_center;
    for (int i = 0; i < centers_.size(); i++) {
      double d = l2_distance(point, centers_[i]);
      if (d < min_distance) {
        best_center = i;
        min_distance = d;
      }
    }
    bool open_new = absl::Bernoulli(
        *gen_, std::min(1.0, std::pow(min_distance, 2) / denominator_prob_));
    if (open_new) {
      create_center(point);
    } else {
      add_point_to_center(point, best_center, min_distance);
    }
  }

  // Returns the estimate of the multiplicities of the centers assigned after a
  // certain time.
  void weighted_centers(const int after_time,
                        std::vector<TimePointPair>* centers,
                        std::vector<double>* weights) {
    centers->clear();
    weights->clear();
    CHECK_EQ(multiplicities_.size(), centers_.size());
    centers->reserve(multiplicities_.size());
    weights->reserve(multiplicities_.size());
    for (int i = 0; i < multiplicities_.size(); i++) {
      centers->push_back(centers_[i]);
      weights->push_back(multiplicities_[i].total_after_time(after_time));
    }
  }

  // Computes a solution of the state of the Meyerson Sketch after time
  // "after_time".
  bool solution(const int after_time, std::vector<TimePointPair>* centers,
                double* cost) const {
    centers->clear();
    *cost = 0.0;

    if (failed_sketch_) {
      centers = nullptr;
      *cost = std::numeric_limits<double>::max();
      return false;
    }

    if (after_time == 0 && precomputed_cost.has_value()) {
      // No need for recomputation.
      CHECK(precomputed_solution.has_value());
      *centers = precomputed_solution.value();
      *cost = precomputed_cost.value();
      return true;
    }

    // Create weighted instance.
    std::vector<std::pair<TimePointPair, double>> instance;
    CHECK_EQ(multiplicities_.size(), centers_.size());
    CHECK_EQ(multiplicities_.size(), costs_sum_dist_.size());
    CHECK_EQ(multiplicities_.size(), costs_sum_sq_dist_.size());
    instance.reserve(multiplicities_.size());
    for (int i = 0; i < multiplicities_.size(); i++) {
      instance.push_back(std::make_pair(
          centers_[i], multiplicities_[i].total_after_time(after_time)));
    }
    double cost_instance = 0;

    // Solve k-means.
    std::vector<int32_t> pos_centers;
    k_means_plus_plus(instance, k_, gen_, &pos_centers, &cost_instance);
    for (const auto& pos : pos_centers) {
      CHECK_LT(pos, instance.size());
      centers->push_back(instance.at(pos).first);
    }
    // Adding the cost of the sketch itset (sum of squared distances)
    for (int i = 0; i < costs_sum_sq_dist_.size(); i++) {
      cost_instance += costs_sum_sq_dist_.at(i).total_after_time(after_time);
    }
    // Adding 2*dist_center_to_sketch(sum dist to sketch center)
    for (int i = 0; i < centers_.size(); i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (const auto& center : *centers) {
        min_distance = std::min(
            min_distance, l2_distance(centers_.at(i).second, center.second));
      }
      cost_instance += 2.0 * min_distance *
                       costs_sum_dist_.at(i).total_after_time(after_time);
    }
    *cost = cost_instance;

    // Update the precomputed solution.
    if (after_time == 0) {
      precomputed_solution.reset();
      precomputed_solution.emplace(*centers);
      precomputed_cost = *cost;
      last_precomputed_0_multiplicities_.clear();
      for (const auto& mult : multiplicities_) {
        last_precomputed_0_multiplicities_.push_back(mult.count_at_0());
      }
      last_precomputed_0_costs_sum_dist_.clear();
      for (const auto& mult : costs_sum_dist_) {
        last_precomputed_0_costs_sum_dist_.push_back(mult.count_at_0());
      }
      last_precomputed_0_costs_sum_sq_dist_.clear();
      for (const auto& mult : costs_sum_sq_dist_) {
        last_precomputed_0_costs_sum_sq_dist_.push_back(mult.count_at_0());
      }
    }

    return true;
  }

  // Compose this sketch as summary_B  with summary_A to the left.
  bool compose_left_solution(const MeyersonSketch& summary_A,
                             int64_t window_begin,
                             std::vector<TimePointPair>* centers,
                             double* cost) const {
    centers->clear();
    if (failed_sketch_ || summary_A.failed_sketch_) {
      centers = nullptr;
      *cost = std::numeric_limits<double>::max();
      return false;
    }

    std::vector<std::pair<TimePointPair, double>> instance;

    // Add all centers in B
    CHECK_EQ(summary_A.multiplicities_.size(), summary_A.centers_.size());
    CHECK_EQ(summary_A.multiplicities_.size(),
             summary_A.costs_sum_dist_.size());
    CHECK_EQ(summary_A.multiplicities_.size(),
             summary_A.costs_sum_sq_dist_.size());
    instance.reserve(summary_A.multiplicities_.size() + multiplicities_.size());
    for (int i = 0; i < summary_A.multiplicities_.size(); i++) {
      instance.push_back(std::make_pair(
          summary_A.centers_.at(i),
          summary_A.multiplicities_.at(i).total_after_time(window_begin)));
    }

    CHECK_EQ(multiplicities_.size(), centers_.size());
    CHECK_EQ(multiplicities_.size(), costs_sum_dist_.size());
    CHECK_EQ(multiplicities_.size(), costs_sum_sq_dist_.size());
    instance.reserve(multiplicities_.size());
    for (int i = 0; i < multiplicities_.size(); i++) {
      instance.push_back(std::make_pair(
          centers_.at(i), multiplicities_.at(i).total_after_time(0)));
    }
    double cost_instance = 0;

    std::vector<int32_t> pos_centers;
    k_means_plus_plus(instance, k_, gen_, &pos_centers, &cost_instance);
    for (const auto& pos : pos_centers) {
      CHECK_LT(pos, instance.size());
      centers->push_back(instance.at(pos).first);
    }
    // Adding the cost of the sketch itself
    for (int i = 0; i < summary_A.costs_sum_sq_dist_.size(); i++) {
      cost_instance +=
          summary_A.costs_sum_sq_dist_.at(i).total_after_time(window_begin);
    }
    for (int i = 0; i < costs_sum_sq_dist_.size(); i++) {
      cost_instance += costs_sum_sq_dist_.at(i).total_after_time(0);
    }
    // Adding the cost of 2*d(sum dist)
    for (int i = 0; i < centers_.size(); i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (const auto& center : *centers) {
        min_distance = std::min(
            min_distance, l2_distance(centers_.at(i).second, center.second));
      }
      cost_instance +=
          2.0 * min_distance * costs_sum_dist_.at(i).total_after_time(0);
    }
    for (int i = 0; i < summary_A.centers_.size(); i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (const auto& center : *centers) {
        min_distance = std::min(
            min_distance,
            l2_distance(summary_A.centers_.at(i).second, center.second));
      }
      cost_instance +=
          2.0 * min_distance *
          summary_A.costs_sum_dist_.at(i).total_after_time(window_begin);
    }
    *cost = cost_instance;
    return true;
  }

 private:
  // Initialize a new center
  void create_center(const TimePointPair& point) {
    if (failed_sketch_ || centers_.size() == max_num_centers_) {
      failed_sketch_ = true;
      return;
    }
    // Invalidate precomputed solution.
    precomputed_cost.reset();

    ++ITEMS_STORED;
    centers_.push_back(point);
    multiplicities_.push_back(ApproxTimeCountKeeper(epsilon_multiplicities_));
    multiplicities_.back().increase_total(point.first, 1);
    costs_sum_dist_.push_back(ApproxTimeCountKeeper(epsilon_multiplicities_));
    costs_sum_dist_.back().increase_total(point.first, 0.0);
    costs_sum_sq_dist_.push_back(
        ApproxTimeCountKeeper(epsilon_multiplicities_));
    costs_sum_sq_dist_.back().increase_total(point.first, 0.0);
  }

  // Assign a point to a center.
  void add_point_to_center(const TimePointPair& point,
                           const int32_t center_position,
                           const double distance) {
    multiplicities_[center_position].increase_total(point.first, 1);
    costs_sum_dist_[center_position].increase_total(point.first, distance);
    costs_sum_sq_dist_[center_position].increase_total(point.first,
                                                       ::std::pow(distance, 2));

    if (precomputed_cost.has_value()) {
      if (multiplicities_.at(center_position).count_at_0() >
              last_precomputed_0_multiplicities_.at(center_position) *
                  (1.0 + kErrorMarginPrecomputedSolution) ||
          costs_sum_dist_.at(center_position).count_at_0() >
              last_precomputed_0_costs_sum_dist_.at(center_position) *
                  (1.0 + kErrorMarginPrecomputedSolution) ||
          costs_sum_sq_dist_.at(center_position).count_at_0() >
              last_precomputed_0_costs_sum_sq_dist_.at(center_position) *
                  (1.0 + kErrorMarginPrecomputedSolution)) {
        // Invalidate precomputed solution.
        precomputed_cost.reset();
      }
    }
  }

  double max_num_centers_;
  // This is the factor used in p = min(dist / denominator_prob_, 1) for
  // inserting a point.
  double denominator_prob_;
  double epsilon_multiplicities_;
  int32_t k_;
  absl::BitGen* gen_;
  bool failed_sketch_;
  vector<TimePointPair> centers_;
  vector<ApproxTimeCountKeeper> multiplicities_;
  vector<ApproxTimeCountKeeper> costs_sum_dist_;
  vector<ApproxTimeCountKeeper> costs_sum_sq_dist_;

  // Used to avoid recomputation if the solution has not significantly changed.
  mutable absl::optional<double> precomputed_cost;
  mutable absl::optional<std::vector<TimePointPair>> precomputed_solution;
  mutable vector<double> last_precomputed_0_multiplicities_;
  mutable vector<double> last_precomputed_0_costs_sum_dist_;
  mutable vector<double> last_precomputed_0_costs_sum_sq_dist_;

  // Margin of error in the precomputed solution.
  static constexpr double kErrorMarginPrecomputedSolution = 0.05;
};

// Implementation of the KMeans summary algorithm for sliding window streams
// using the Meyerson sketch.
class KMeansSummary : public SummaryAlg {
 public:
  KMeansSummary(int64_t window, int32_t k, double optimum_upperbound_guess,
                absl::BitGen* gen)
      : SummaryAlg(window, k, optimum_upperbound_guess, gen) {
    max_sketch_size_ = std::pow(2, 2 * 2 + 1) * 4. * k *
                       (1. + std::log(window * 3)) * (1.0 + 1. / 0.5);
    distance_denominator_ =
        (optimum_upperbound_guess) / (k * (1. + std::log(window * 3)));
    // sketch_number_ = std::log(3 * 1. / error_prob);
    gen_ = gen;
    k_ = k;
    reset_impl();
  }

  virtual ~KMeansSummary() {}

  void solution(vector<TimePointPair>* centers_solution,
                double* cost_solution) const override {
    centers_solution->clear();
    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<TimePointPair> centers;
      bool ok = sketches_.at(i).solution(/*after_time=*/0, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }
    sketches_.at(best_pos).solution(/* after_time=*/0, centers_solution,
                                    cost_solution);
  }

  void solution_left_composition(const SummaryAlg& summary_A,
                                 int64_t window_begin,
                                 vector<TimePointPair>* sol,
                                 double* value) const override {
    sol->clear();
    const KMeansSummary& summary_A_type =
        dynamic_cast<const KMeansSummary&>(summary_A);

    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    CHECK_EQ(summary_A_type.sketches_.size(), sketches_.size());

    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<TimePointPair> centers;
      bool ok = sketches_.at(i).compose_left_solution(
          summary_A_type.sketches_.at(i), window_begin, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }

    sketches_.at(best_pos).compose_left_solution(
        summary_A_type.sketches_.at(best_pos), window_begin, sol, value);
  }

  void solution_right_composition_with_empty(
      int64_t window_begin, vector<TimePointPair>* centers_solution,
      double* cost_solution) const override {
    double min_cost = std::numeric_limits<double>::max();
    int best_pos = 0;
    for (int i = 0; i < kSketchNumber; i++) {
      double cost;
      vector<TimePointPair> centers;
      bool ok = sketches_.at(i).solution(window_begin, &centers, &cost);
      if (ok && cost < min_cost) {
        min_cost = cost;
        best_pos = i;
      }
    }
    sketches_.at(best_pos).solution(window_begin, centers_solution,
                                    cost_solution);
  }

 protected:
  void process_point_impl(const int64_t time,
                          const std::vector<double>& point) override {
    for (auto& sketch : sketches_) {
      sketch.add_point(std::make_pair(time, point));
    }
  }

  void reset_impl() override {
    sketches_.clear();
    sketches_.reserve(kSketchNumber);
    for (int i = 0; i < kSketchNumber; i++) {
      MeyersonSketch sketch(max_sketch_size_, distance_denominator_,
                            kEpsilonMult, k_, gen_);
      sketches_.push_back(sketch);
    }
  }

 private:
  int32_t max_sketch_size_;
  double distance_denominator_;
  int32_t k_;
  absl::BitGen* gen_;

  std::vector<MeyersonSketch> sketches_;

  // Epsilon factor in the estimation of the multiplicities.
  static constexpr double kEpsilonMult = 0.01;
  // Number of sketches used. As stated in the paper, we use a single sketch as
  // in practice is sufficient to get good results.
  static constexpr int32_t kSketchNumber = 1;
};

// Baseline algorithm that stores the entire window and as solution, uses a
// random either runs kmeans++ on the whole dataset or a sample. Notice that
// this class does not update the ITEMS_STORED count which is only used to
// evaluate the main algorithm. However, calling this function affects the total
// number of l2 distance calls.
class BaselineKmeansOverSampleWindow {
 public:
  // window is the sliding window size.
  // k is the number of centers.
  // number_tries is the number of times kmeans++ is used for the whole window
  // baseline.
  // gen is a random source.
  BaselineKmeansOverSampleWindow(int64_t window, int32_t k,
                                 int32_t number_tries, absl::BitGen* gen)
      : window_(window),
        k_(k),
        number_tries_(number_tries),
        gen_(gen),
        window_handler_(window) {
    CHECK_GT(k, 0);
    CHECK_GT(window_, 1);
  }
  // Disllow copy and assign.
  BaselineKmeansOverSampleWindow(BaselineKmeansOverSampleWindow const&) =
      delete;
  void operator=(BaselineKmeansOverSampleWindow const& x) = delete;

  // Process a data point.
  void process_point(const int64_t time, const std::vector<double>& point) {
    window_handler_.next();
    CHECK_EQ(time, window_handler_.curr_time());
    deque_.push_back(std::make_pair(time, point));
    while (deque_.front().first != window_handler_.begin_window()) {
      deque_.pop_front();
    }
    CHECK_EQ(deque_.size(),
             window_handler_.curr_time() - window_handler_.begin_window() + 1);
  }

  // Using the whole window compute a solution and its cost.
  void solution(std::vector<TimePointPair>* solution, double* sol_cost) {
    solution->clear();
    std::vector<std::pair<TimePointPair, double>> instance;
    for (const auto& point : deque_) {
      instance.push_back(std::make_pair(point, 1.0));
    }
    std::vector<int32_t> best_centers;
    double min_cost = std::numeric_limits<double>::max();
    for (int i = 0; i < number_tries_; i++) {
      std::vector<int32_t> centers;
      double cost;
      k_means_plus_plus(instance, k_, gen_, &centers, &cost);
      if (cost < min_cost) {
        min_cost = cost;
        best_centers = centers;
      }
    }
    std::vector<TimePointPair> sol;
    for (auto pos : best_centers) {
      CHECK_LT(pos, instance.size());
      solution->push_back(instance.at(pos).first);
    }
    *sol_cost = min_cost;
  }

  // Sample u.a.r sample_size points from the window and obtain solution and its
  // cost using them.
  void solution_subsampling(int64_t sample_size,
                            std::vector<TimePointPair>* solution,
                            double* solution_cost) {
    solution->clear();
    sample_size = std::min(sample_size, (int64_t)deque_.size());

    std::vector<std::pair<TimePointPair, double>> instance;
    std::vector<TimePointPair> all_points;
    for (const auto& point : deque_) {
      instance.push_back(std::make_pair(point, 1.0));
      all_points.push_back(point);
    }
    std::shuffle(instance.begin(), instance.end(), *gen_);
    instance.resize(sample_size);
    std::vector<int32_t> centers;
    double ingored_cost;
    k_means_plus_plus(instance, k_, gen_, &centers, &ingored_cost);

    for (const auto& pos : centers) {
      solution->push_back(instance.at(pos).first);
    }
    *solution_cost = cost_solution(all_points, *solution);
  }

  std::vector<TimePointPair> points_window() {
    std::vector<TimePointPair> copy;
    for (const auto& e : deque_) {
      copy.push_back(e);
    }
    return copy;
  }

 private:
  const int64_t window_;
  const int32_t k_;
  const int32_t number_tries_;
  absl::BitGen* gen_;
  SlidingWindowHandler window_handler_;
  std::deque<TimePointPair> deque_;
};

}  //  namespace sliding_window

#endif  // SLIDING_WINDOW_CLUSTERING_MEYERSON_ALGORITHM_H_
