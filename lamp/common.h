// Copyright 2025 The Google Research Authors.
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

#ifndef LAMP_COMMON_H
#define LAMP_COMMON_H

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Common types.

typedef std::vector<std::string> StringTrail;
struct SplitStringTrail {
  StringTrail trail;
  int train_size;  // This long prefix is the training data, or all if < 0
};
typedef std::vector<SplitStringTrail> SplitStringTrails;

// Each location is assigned an int identifier.
typedef std::vector<int> Trail;
typedef std::vector<Trail> Trails;
typedef std::vector<double> WeightVector;

// For parameters and derivates of locations and transition matrices.
typedef std::unordered_map<int, double> SparseVector;
typedef std::vector<SparseVector> SparseMatrix;

// Used in cross-validation. Refers to trails[beging_index..end_index) where the
// interval is mod trails.size()
class TrailsSlice {
 public:
  TrailsSlice(const Trails& trails, int begin_index, int end_index,
              bool is_test)
      : is_test_(is_test),
        trails_(trails),
        begin_index_(begin_index),
        end_index_(end_index),
        wraps_around_(end_index < begin_index) {}
  // All trails
  TrailsSlice(const Trails& trails, bool is_test)
      : is_test_(is_test),
        trails_(trails),
        begin_index_(0),
        end_index_(trails.size()),
        wraps_around_(false) {}

  class const_iterator {
   public:
    const_iterator(const Trails& trails, int index, bool should_wrap_around)
        : it_(trails.cbegin() + index),
          trails_begin_(trails.cbegin()),
          trails_end_(trails.cend()),
          should_wrap_around_(should_wrap_around) {}
    const_iterator operator++() {
      ++it_;
      if (it_ == trails_end_ && should_wrap_around_) {
        it_ = trails_begin_;
      }
      return *this;
    }
    bool operator!=(const const_iterator& other) { return it_ != other.it_; }
    const Trail& operator*() const { return *it_; }

   private:
    Trails::const_iterator it_;
    const Trails::const_iterator trails_begin_, trails_end_;
    bool should_wrap_around_;
  };

  const_iterator begin() const {
    return const_iterator(trails_, begin_index_, wraps_around_);
  }
  const_iterator end() const {
    return const_iterator(trails_, end_index_, wraps_around_);
  }

  // Used by PrintEval only.
  const bool is_test_;

 private:
  const Trails& trails_;
  const int begin_index_, end_index_;
  const bool wraps_around_;
};

std::string ToString(const Trail& trail);

// Returns value of a coordinate or 0 if not found. Does not smooth!
inline double GetOrZero(const SparseVector& values, int col) {
  const auto it = values.find(col);
  return it == values.end() ? 0 : it->second;
}

inline double GetSmoothedOld(const SparseVector& row_values, int num_locations,
                             int col) {
  // Avoid log(0) by assigning at least this much probability to all row ->
  // col transitions.
  double epsilon = 1e-3 / num_locations;
  //    double epsilon = 1e-5 / num_locations;
  return (1 - epsilon) * GetOrZero(row_values, col) + epsilon;
}

// LAMP model
struct Model {
  // Dense vector of unigram frequencies, used for smoothing
  double sum_freq;
  WeightVector item_frequencies;
  SparseMatrix transition_probs;
  WeightVector weights;

  // Returns transition_probs[row][col] smoothed to 0<x<=1.
  // Defined in the header for efficieny.
  //
  double GetProb(int row, int col, double* prob_coeff) const {
    // [row] without bound checking is marginally faster than .at(row) which
    // throws an exception if the index is out of range.
    // return GetSmoothedOld(transition_probs[row], transition_probs.size(),
    // col);

    // Smoothing.
    const double eps_unigram = 0.25, eps_bigram = 2.0;
    // Turns off smoothing effectively for testing with toy data.
    // const double eps_unigram = 1e6, eps_bigram = 1e-6;
    const double smoothed_unigram_prob =
        (item_frequencies[col] + eps_unigram) /
        (sum_freq + item_frequencies.size() * eps_unigram);
    const double unigram_weight =
        eps_bigram / (item_frequencies[row] + eps_bigram);
    *prob_coeff = 1.0 - unigram_weight;
    return GetOrZero(transition_probs[row], col) * (*prob_coeff) +
           unigram_weight * smoothed_unigram_prob;
  }

  double GetProb(int row, int col) const {
    double unused_prob_coeff;
    return GetProb(row, col, &unused_prob_coeff);
  }
};

// Transition matrix for 2nd and higher order models.
struct NGramLess {
  bool operator()(const std::vector<int>& x, const std::vector<int>& y) const {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  }
};

typedef std::map<std::vector<int>, SparseVector, NGramLess> NGramMatrix;
typedef std::map<std::vector<int>, int, NGramLess> NGramFrequency;

struct NGramModel {
  int num_locations;
  // stores counts of ngrams grouped by all but the last item, i.e. count of abc
  // is counts[{a,b}][c].
  NGramMatrix counts;
  // sum of ngram's row in counts ~ frequency of each n-gram
  NGramFrequency sum_counts;
  int order;
};

// Kneyser-Ney n-gram model
struct KneyserNeyModel {
  int num_locations;
  // stores counts of ngrams grouped by all but the last item, i.e. count of abc
  // is counts[{a,b}][c].
  NGramMatrix counts;
  // sum of ngram's row in counts ~ frequency of each n-gram
  NGramFrequency sum_counts;
  int order;
  // N_{1+} in the paper. Maps a key ngram to the unique number of x items such
  // that x followed by key ngram occured at least once.
  NGramFrequency num_antecedents;
  // sum N_{1+} in the paper. Maps a key ngram to the unique number x, y item
  // comnbinatios such that x followed by key followed by y occured at least
  // once. The same as sum_y num_antecedents[key followed y].
  NGramFrequency middle_counts;
};

// Keys set are "loglik", "perplexity", "num_params";
// Stored in key-value format so that it's easy to compute mean and variance
// over crossvalidation folds.
typedef std::map<std::string, double> MetricsMap;
std::string ToTSVString(const MetricsMap& metrics);

struct Eval {
  MetricsMap metrics;

  double loglik() const { return metrics.at("loglik"); }
  double perplexity() const { return metrics.at("perplexity"); }
  double accuracy() const { return metrics.at("accuracy"); }
  double num_params() const { return metrics.at("num_params"); }

  // Only set for LAMPs.
  WeightVector weights;
};

struct Algorithm {
  std::string name;
  int order;
  int iter;  // 2 * i (+1), where is is number of iterations in alternating
             // optimization
  bool is_final_iter;
  bool is_test;

  bool operator<(const Algorithm& rhs) const {
    return std::tie(name, order, iter, is_final_iter, is_test) <
           std::tie(rhs.name, rhs.order, rhs.iter, rhs.is_final_iter,
                    rhs.is_test);
  }
};

typedef std::map<Algorithm, std::vector<Eval>> EvalMap;

// Prints error message with errno and exits.
void die(const std::string& message);
void die(const std::string& msg_prefix, const std::string& msg_postfix);

void die_if(bool is_error, const std::string& message);
void die_if(bool is_error, const std::string& msg_prefix,
            const std::string& msg_postfix);
void check_prob(double prob, const char* message_postfix);

double Sum(const WeightVector& v);
// v = a * v;
void Scale(double a, WeightVector* v);
void Normalize(WeightVector* v);
// L_{infinity} norm of x - y.
double MaxAbsDifference(const WeightVector& x, const WeightVector& y);
std::string ToString(const WeightVector& weights);

double Sum(const SparseVector& v);
// v = a * v;
void Scale(double a, SparseVector* v);
// Makes L1-norm unit.
void Normalize(SparseVector* v);
std::string ToString(const SparseVector& v);

// Multiplies each element of transitions with a.
void Scale(double a, SparseMatrix* transitions);
// Normalizes each row.
void Normalize(SparseMatrix* transitions);
std::string ToString(const SparseMatrix& transitions);

SparseMatrix ComputeEmpiricalTransitionMatrix(const TrailsSlice& trails,
                                              int num_locations,
                                              WeightVector* item_frequencies);

// Uses num_locations for smoothing 0 probabilities.
double LogLikelihood(const Trail& trail, const Model& model);
double LogLikelihood(const TrailsSlice& trails, const Model& model);

void Normalize(NGramMatrix* transitions);
NGramModel CountNGrams(const TrailsSlice& trails, int max_order);

double LogLikelihood(const Trail& trail, const NGramModel& model);
double LogLikelihood(const TrailsSlice& trails, const NGramModel& model);

KneyserNeyModel CountKneyserNey(const TrailsSlice& trails, int max_order);
double LogLikelihood(const Trail& trail, const KneyserNeyModel& model,
                     double discount);
double LogLikelihood(const TrailsSlice& trails, const KneyserNeyModel& model,
                     double discount);

// Returns the total number of transitions in trails = sum length trails - num
// trails.
int NumTransitions(const TrailsSlice& trails);
int NumTransitions(const Trails& trails);
int NumTransitions(const SplitStringTrails& trails);

// Computes perplexity (=1.0/(geometric mean of transition probabilities)).
double Perplexity(double log_likelihood, int num_transitions);

// Print loglik and perplexity.
void PrintEval(const char* print_prefix, double loglik, int num_transitions);
void PrintEval(const char* print_prefix, double loglik, int num_transitions,
               const WeightVector& weights);
/// Print loglik, perplexity and model weights.
void PrintEval(const char* print_prefix, const Eval& eval, bool is_test,
               bool print_num_params = false);
Eval PrintEval(const TrailsSlice& trails, const Model& model,
               const char* print_prefix, bool print_num_params = false,
               bool should_print = true);
Eval Evaluate(const TrailsSlice& trails, const Model& model);

inline void PrintAndAddEval(const char* alg, int order, int iter,
                            bool is_final_iter, const TrailsSlice& trails,
                            const Model& model, EvalMap* eval_map) {
  Eval eval = PrintEval(trails, model, alg, true, is_final_iter);
  (*eval_map)[{alg, order, iter, is_final_iter, trails.is_test_}].push_back(
      eval);
}

// Not used currently.
// Print loglik, perplexity and model transitions.
// void PrintEval(const char* print_prefix, double loglik, int
// num_transitions,
//               const SparseMatrix& transition_probs);

// Return trail[base_index - back_shitf] or trails[0] if the
// former is out of
// bounds.
inline int Get(const Trail& trail, int base_index, int back_shift) {
  int k = std::max(base_index - back_shift, 0);
  return trail[k];
}

#endif
