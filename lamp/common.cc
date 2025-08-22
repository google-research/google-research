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

#include "common.h"  // NOLINT(build/include)

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>

// NOLINTBEGIN(runtime/printf)

void die(const std::string& message) {
  if (errno != 0) {
    perror(message.c_str());
  } else {
    fprintf(stderr, "%s\n", message.c_str());
  }
  exit(1);
}

void die(const std::string& msg_prefix, const std::string& msg_postfix) {
  die(msg_prefix + " " + msg_postfix);
}

void die_if(bool is_error, const std::string& message) {
  if (is_error) {
    die(message);
  }
}

void die_if(bool is_error, const std::string& msg_prefix,
            const std::string& msg_postfix) {
  if (is_error) {
    die(msg_prefix, msg_postfix);
  }
}

void check_prob(double prob, const char* message_postfix) {
  if (!std::isfinite(prob) || prob < 0 || prob > 1 + 1e-10) {
    char buffer[256];
    sprintf(buffer, "Prob %g out of bounds in %s", prob, message_postfix);
    die(buffer);
  }
}

std::string ToString(const Trail& trail) {
  std::string ret;
  for (int item : trail) {
    char buffer[128];
    sprintf(buffer, "%d,", item);
    ret.append(buffer);
  }
  return ret;
}

double Sum(const WeightVector& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

void Scale(double a, WeightVector* v) {
  for (double& value : *v) {
    value *= a;
  }
}

void Normalize(WeightVector* v) { Scale(1.0 / Sum(*v), v); }

double MaxAbsDifference(const WeightVector& x, const WeightVector& y) {
  die_if(x.size() != y.size(), "Mismatched vector sizes in MaxAbsDifference");
  double ret = 0;
  for (size_t i = 0; i < x.size(); i++) {
    ret = std::max(std::fabs(x[i] - y[i]), ret);
  }
  return ret;
}

double Sum(const SparseVector& v) {
  double sum = 0;
  for (const auto& item_and_value : v) {
    die_if(item_and_value.second < 0, "Negative sparse vector entry");
    sum += item_and_value.second;
  }
  die_if(sum <= 0, "Non-positive sparse vector norm");
  return sum;
}

std::string ToString(const WeightVector& weights) {
  std::string ret = "[";
  for (size_t i = 0; i < weights.size(); i++) {
    char buffer[128];
    sprintf(buffer, "%g", weights[i]);
    ret += buffer;
    if (i + 1 != weights.size()) {
      ret += ", ";
    }
  }
  ret += "]";
  return ret;
}

void Scale(double a, SparseVector* v) {
  for (auto& item_and_value : *v) {
    item_and_value.second *= a;
  }
}

void Normalize(SparseVector* v) { Scale(1.0 / Sum(*v), v); }

void Normalize(SparseMatrix* transitions) {
  for (SparseVector& row : *transitions) {
    if (!row.empty()) {
      Normalize(&row);
    }
  }
}

void Scale(double a, SparseMatrix* transitions) {
  for (SparseVector& row : *transitions) {
    Scale(a, &row);
  }
}

SparseMatrix ComputeEmpiricalTransitionMatrix(const TrailsSlice& trails,
                                              int num_locations,
                                              WeightVector* item_frequencies) {
  SparseMatrix transitions(num_locations);
  item_frequencies->clear();
  item_frequencies->resize(num_locations);
  for (const Trail& trail : trails) {
    for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
      (transitions.at(trail[i]))[trail[i + 1]] += 1;
      item_frequencies->at(trail[i]) += 1;
    }
    if (!trail.empty()) {
      item_frequencies->at(trail.back()) += 1;
    }
  }
  Normalize(&transitions);
  return transitions;
}

double LogLikelihood(const Trail& trail, const Model& model) {
  double ret = 0;
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    double prob = 0;
    int col = trail[i + 1];
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      prob += model.weights[j] * model.GetProb(row, col);
    }
    if (prob > 1 + 1e-10) {
      for (size_t j = 0; j < model.weights.size(); j++) {
        int row = Get(trail, i, j);
        printf("LogLikelihood: j %zu row %d col %d weight %f GetProb %f\n", j,
               row, col, model.weights[j], model.GetProb(row, col));
      }
      printf("LogLikelihood large prob: %f\n", prob);
    }
    check_prob(prob, "transition probability");
    ret += log(prob);
  }
  return ret;
}

double LogLikelihood(const TrailsSlice& trails, const Model& model) {
  double ret = 0;
  for (const Trail& trail : trails) {
    ret += LogLikelihood(trail, model);
  }
  return ret;
}

struct ColAndProb {
  int col;
  double prob;

  bool operator<(const ColAndProb& rhs) const { return prob < rhs.prob; }
};
typedef std::vector<ColAndProb> SortedProbVector;
typedef std::vector<SortedProbVector> SortedTransitionMatrix;

// Closer to brute-force and sloooow.
int NumCorrectPredictionsOld(
    const Trail& trail, const Model& model,
    const SortedTransitionMatrix& sorted_transition_probs) {
  int num_correct = 0;
  SparseVector predicted_probs;
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    const double kFudge = 1e-12;
    // Add tiny fudge factor to avoid any potential issues with comparing a
    // double to itself (perhaps computed in a slightly different way (though
    // we are adding numbers in the same order)).
    double prob_of_actual = kFudge;
    int col = trail[i + 1];
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      prob_of_actual += model.weights[j] * model.GetProb(row, col);
    }
    bool prob_of_actual_is_max = true;
    predicted_probs.clear();
    for (size_t j = 0; j < model.weights.size() && prob_of_actual_is_max; j++) {
      int row = Get(trail, i, j);
      for (const ColAndProb& col_and_prob : sorted_transition_probs[row]) {
        predicted_probs[col_and_prob.col] +=
            model.weights[j] * col_and_prob.prob;
        // TODO(stamas): We are cheating here a bit as predicted_probs does not
        // contain terms arising from smoothing if a row->col transition is not
        // present in the transition matrix.
        if (predicted_probs[col_and_prob.col] > prob_of_actual) {
          prob_of_actual_is_max = false;
          break;
        }
      }
    }
    if (prob_of_actual_is_max) {
      num_correct++;
    }
  }
  return num_correct;
}

struct IndexAndProb {
  size_t weight_index;  // which weight it corresponds to.
  size_t position;      // current position in the sorted transitions of row.
  double prob;          // transition prob * weight

  bool operator<(const IndexAndProb& rhs) const { return prob < rhs.prob; }
};

// Fagin's Threshold Algorithm.
double NumCorrectPredictions(
    const Trail& trail, const Model& model,
    const SortedTransitionMatrix& sorted_transition_probs,
    int* num_predictions) {
  double num_correct = 0;
  std::unordered_set<int> evaluated_items;
  std::vector<int> rows(model.weights.size());
  std::vector<IndexAndProb> heap;
  // printf("Trail: %s\n", ToString(trail).c_str());
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    int col = trail[i + 1];
    // Add tiny fudge factor to avoid any potential issues with comparing a
    // double to itself (perhaps computed in a slightly different way (though
    // we are adding numbers in the same order)).
    const double kFudge = 1e-12;
    double prob_of_actual = (-1) * kFudge;
    evaluated_items.clear();
    heap.clear();
    *num_predictions += 1;
    double upper_bound = 0.0;  // Upper bound for unseen elements.
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      rows[j] = row;
      if (!sorted_transition_probs[row].empty()) {
        double partial_prob =
            model.weights[j] * sorted_transition_probs[row].front().prob;
        upper_bound += partial_prob;
        heap.push_back({j, 0, partial_prob});
      }
      prob_of_actual += model.weights[j] * model.GetProb(row, col);
    }
    // printf("index %d curr %d next %d prob of next %g\n",
    //          i, trail[i], col, prob_of_actual);
    std::make_heap(heap.begin(), heap.end());
    bool prob_of_actual_is_max = true;
    int num_equal = 1;
    while (upper_bound > prob_of_actual && !heap.empty()) {
      IndexAndProb head = heap.front();
      const SortedProbVector& sorted_probs =
          sorted_transition_probs[rows[head.weight_index]];
      int candidate = sorted_probs[head.position].col;
      // printf("upper bound %g candidate %d\n", upper_bound, candidate);
      if (candidate != col &&
          evaluated_items.find(candidate) == evaluated_items.end()) {
        evaluated_items.insert(candidate);
        double prob_of_candidate = 0;
        for (size_t j = 0; j < model.weights.size(); j++) {
          int row = rows[j];
          prob_of_candidate += model.weights[j] * model.GetProb(row, candidate);
        }
        // printf("prob of candidate %g\n", prob_of_candidate);
        if (prob_of_candidate > prob_of_actual + 2 * kFudge) {
          prob_of_actual_is_max = false;
          break;
        }
        if (prob_of_candidate > prob_of_actual) {
          num_equal++;
        }
      }
      // Update max heap and upper bound.
      upper_bound -= head.prob;
      std::pop_heap(heap.begin(), heap.end());
      heap.pop_back();
      head.position++;
      if (head.position < sorted_probs.size()) {
        head.prob =
            model.weights[head.weight_index] * sorted_probs[head.position].prob;
        heap.push_back(head);
        std::push_heap(heap.begin(), heap.end());
        upper_bound += head.prob;
      }
    }
    if (prob_of_actual_is_max) {
      // printf("Correct: num equal %d\n", num_equal);
      num_correct += 1.0 / num_equal;
    }
  }
  return num_correct;
}

double Accuracy(const TrailsSlice& trails, const Model& model) {
  double num_correct_predictions = 0;
  // Precompute smoothing for optimized access.
  SortedTransitionMatrix sorted_transition_probs;
  sorted_transition_probs.resize(model.transition_probs.size());
  for (size_t row = 0; row < model.transition_probs.size(); row++) {
    sorted_transition_probs[row].reserve(model.transition_probs[row].size());
    for (const auto& col_and_value : model.transition_probs[row]) {
      sorted_transition_probs[row].push_back(
          {col_and_value.first, model.GetProb(row, col_and_value.first)});
    }
    // Sort descending.
    std::sort(sorted_transition_probs[row].rbegin(),
              sorted_transition_probs[row].rend());
  }
  int num_predictions = 0;
  for (const Trail& trail : trails) {
    num_correct_predictions += NumCorrectPredictions(
        trail, model, sorted_transition_probs, &num_predictions);
  }
  return num_correct_predictions / num_predictions;
}

int NumTransitions(const TrailsSlice& trails) {
  int ret = 0;
  for (const Trail& trail : trails) {
    die_if(trail.empty(), "Empty trail");
    ret += trail.size() - 1;
  }
  return ret;
}

int NumTransitions(const Trails& trails) {
  return NumTransitions(TrailsSlice(trails, false));
}

int NumTransitions(const SplitStringTrails& trails) {
  int ret = 0;
  for (const SplitStringTrail& trail : trails) {
    die_if(trail.trail.empty(), "Empty string trail");
    ret += trail.trail.size() - 1;
  }
  return ret;
}

double Perplexity(double log_likelihood, int num_transitions) {
  return exp((-1) * log_likelihood / num_transitions);
}

void PruneAndNormalize(NGramMatrix* transitions) {
  for (auto it = transitions->begin(); it != transitions->end();
       /* no increment */) {
    double frequency = Sum(it->second);
    // Optionally drop context n-grams with low counts
    if (!it->first.empty() && frequency < 5) {
      transitions->erase(it++);
    } else {
      Scale(1.0 / frequency, &it->second);  // normalize
      ++it;
    }
  }
}

std::vector<int> GetContext(const Trail& trail, int base_index, int order) {
  std::vector<int> context(order);
  for (int j = order - 1; j >= 0; j--) {
    context[j] = Get(trail, base_index, j);
  }
  return context;
}

NGramModel CountNGrams(const TrailsSlice& trails, int max_order) {
  die_if(max_order < 1, "NGram order < 1");
  NGramModel model;
  model.order = max_order;
  for (const Trail& trail : trails) {
    for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
      std::vector<int> context = GetContext(trail, i, max_order);
      do {
        model.counts[context][trail[i + 1]] += 1;
        model.sum_counts[context] += 1;
        context.pop_back();
      } while (!context.empty());
      model.counts[{}][trail[i + 1]] += 1;
      model.sum_counts[context] += 1;
    }
  }
  // PruneAndNormalize(&counts);
  return model;
}

double GetSmoothedNGram(const SparseVector& counts, int sum_counts,
                        int num_locations, int col) {
  double epsilon = 1e-3 / num_locations;
  return (1 - epsilon) * GetOrZero(counts, col) /
             static_cast<double>(sum_counts) +
         epsilon;
}

double LogLikelihood(const Trail& trail, const NGramModel& model) {
  double ret = 0;
  // printf("NGram LL of trail %s\n", ToString(trail).c_str());
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    int col = trail[i + 1];
    std::vector<int> context = GetContext(trail, i, model.order);
    double prob = -1.0;
    // Naive backoff to a context seen, used when test data != train data.
    // TODO(stamas): Implement some better, standard language modeling back-off:
    // Laplace (add 1), Katz, or Kneser-Ney. And then have something comparable
    // in LAMP models as well.
    context.push_back(-1);  // dummy item to get the loop started.
    do {
      context.pop_back();
      const auto it = model.counts.find(context);
      if (it != model.counts.end()) {
        prob = GetSmoothedNGram(it->second, model.sum_counts.at(context),
                                model.num_locations, col);
        break;
      }
    } while (!context.empty());
    check_prob(prob, "n-gram transition probability");
    // printf("NGram LL %d %g\n", i, prob);
    ret += log(prob);
  }

  return ret;
}

double LogLikelihood(const TrailsSlice& trails, const NGramModel& model) {
  double ret = 0;
  for (const Trail& trail : trails) {
    ret += LogLikelihood(trail, model);
  }
  return ret;
}

void Print(const NGramFrequency& freqs, const char* name) {
  printf("%s:\n", name);
  for (const auto& it : freqs) {
    printf("%s -> %d\n", ToString(it.first).c_str(), it.second);
  }
}

KneyserNeyModel CountKneyserNey(const TrailsSlice& trails, int max_order) {
  die_if(max_order < 1, "KN-NGram order < 1");
  NGramModel ngram_model = CountNGrams(trails, max_order);
  KneyserNeyModel model;
  model.order = max_order;
  model.counts.swap(ngram_model.counts);
  model.sum_counts.swap(ngram_model.sum_counts);
  std::set<std::vector<int>, NGramLess> unique_ngrams;
  for (const Trail& trail : trails) {
    for (int i = 1; i < static_cast<int>(trail.size()); i++) {
      std::vector<int> ngram = GetContext(trail, i, max_order + 1);
      do {
        unique_ngrams.insert(ngram);
        ngram.pop_back();  // stored reversed.
      } while (ngram.size() >= 2);
    }
  }
  for (std::vector<int> ngram : unique_ngrams) {
    ngram.pop_back();
    model.num_antecedents[ngram]++;
    ngram.erase(ngram.begin());
    model.middle_counts[ngram]++;
  }
  //   Print(model.num_antecedents, "num_antecedents");
  //  Print(model.middle_counts, "middle_counts");
  return model;
}

// smoothed estimate of ngram.front given rest
double ProbKNSmooth(const KneyserNeyModel& model, const std::vector<int>& ngram,
                    double discount) {
  // printf("Begin KNSmooth ngram %s\n", ToString(ngram).c_str());
  if (ngram.empty()) {
    // printf("End KNSmooth %g\n", 1.0 / model.num_locations);
    return 1.0 / model.num_locations;
  }
  std::vector<int> context = ngram;
  context.erase(context.begin());
  std::vector<int> shorter_ngram = ngram;
  shorter_ngram.pop_back();
  double prob = 1.0 / model.middle_counts.at(context);
  // printf("KNSmooth Found middle counts context %s -> %g\n",
  // ToString(context).c_str(), 1.0/prob);
  const auto it =
      model.num_antecedents.find(ngram);  // May not be found for test ngrams
  int num_ante = it == model.num_antecedents.end() ? 0 : it->second;
  // printf("KNSmooth num antecedents %s -> %d\n", ToString(ngram).c_str(),
  // num_ante);
  // printf("KNSmooth Found num uniq contx %s -> %zu\n",
  // ToString(context).c_str(), model.counts.at(context).size());
  prob *= std::max(num_ante - discount, 0.0) +
          discount * model.counts.at(context).size() *
              ProbKNSmooth(model, shorter_ngram, discount);
  // printf("End KNSmooth %g\n", prob);
  return prob;
}

double LogLikelihood(const Trail& trail, const KneyserNeyModel& model,
                     double discount) {
  double ret = 0;
  // printf("\nKN LL of trail %s\n", ToString(trail).c_str());
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    std::vector<int> context = GetContext(trail, i, model.order);
    double prob = -1.0;
    // Naive backoff to a context seen, used when test data != train data, what
    // else can we do?
    context.push_back(-1);  // dummy item to get the loop started.
    int col = trail[i + 1];
    do {
      // printf("KN LL i %d col %d context %s\n", i, col,
      // ToString(context).c_str());
      context.pop_back();
      const auto it = model.counts.find(context);
      if (it != model.counts.end()) {
        if (model.sum_counts.find(context) == model.sum_counts.end()) {
          printf("Couldn't find sum_counts of %s\n", ToString(context).c_str());
        }

        // printf("Found counts of context %s\n", ToString(context).c_str());
        prob = 1.0 / model.sum_counts.at(context);
        // printf("sum counts %d\n", model.sum_counts.at(context));
        // ngram with col at head and 1 shorter context
        std::vector<int> ngram = context;
        ngram.insert(ngram.begin(), col);
        ngram.pop_back();
        // printf("Count %g unix ctx %zu\n", GetOrZero(it->second, col),
        // it->second.size());
        prob *=
            std::max(GetOrZero(it->second, col) - discount, 0.0) +
            discount * it->second.size() * ProbKNSmooth(model, ngram, discount);
        break;
      }
    } while (!context.empty());
    check_prob(prob, "Kneyser-Ney transition probability");
    // printf("KN LL %d %g\n", i, prob);
    ret += log(prob);
  }
  return ret;
}

double LogLikelihood(const TrailsSlice& trails, const KneyserNeyModel& model,
                     double discount) {
  double ret = 0;
  for (const Trail& trail : trails) {
    ret += LogLikelihood(trail, model, discount);
  }
  return ret;
}

void PrintEval(const char* print_prefix, double loglik, int num_transitions,
               const WeightVector& weights) {
  // 13 characters match the width of 'NGram order=1' prefixes to align
  // log-likelihood and perplexity.
  printf("%-13s log-likelihood %f perplexity %f weights=%s\n", print_prefix,
         loglik, Perplexity(loglik, num_transitions),
         ToString(weights).c_str());
}

void PrintEval(const char* print_prefix, const Eval& eval, bool is_test,
               bool print_num_params) {
  // 13 characters match the width of 'NGram order=1' prefixes to align
  // log-likelihood and perplexity.
  printf("%-13s %s log-likelihood %f perplexity %f accuracy %f weights=%s",
         print_prefix, is_test ? "test" : "train", eval.loglik(),
         eval.perplexity(), eval.accuracy(), ToString(eval.weights).c_str());
  if (print_num_params) {
    printf(" num params=%g", eval.num_params());
  }
  printf("\n");
}

Eval Evaluate(const TrailsSlice& trails, const Model& model) {
  Eval eval;
  double loglik = LogLikelihood(trails, model);
  eval.metrics["loglik"] = loglik;
  eval.metrics["perplexity"] = Perplexity(loglik, NumTransitions(trails));
  eval.metrics["accuracy"] = Accuracy(trails, model);
  int num_params = 0;
  for (const SparseVector& t : model.transition_probs) {
    num_params += t.size();
  }
  eval.metrics["num_params"] = num_params;
  eval.weights = model.weights;
  return eval;
}

Eval PrintEval(const TrailsSlice& trails, const Model& model,
               const char* print_prefix, bool print_num_params,
               bool should_print) {
  Eval eval = Evaluate(trails, model);
  if (should_print) {
    PrintEval(print_prefix, eval, trails.is_test_, print_num_params);
  }
  return eval;
}

std::string ToString(const SparseVector& v) {
  std::string ret;
  char buffer[128];
  std::vector<int> columns;
  for (const auto& col_and_value : v) {
    columns.push_back(col_and_value.first);
  }
  std::sort(columns.begin(), columns.end());
  for (int column : columns) {
    sprintf(buffer, "\t%d:%g", column, v.at(column));
    ret.append(buffer);
  }
  return ret;
}

std::string ToString(const SparseMatrix& transitions) {
  std::string ret;
  for (size_t row = 0; row < transitions.size(); row++) {
    char buffer[128];
    sprintf(buffer, "%zu ->", row);
    ret.append(buffer);
    ret.append(ToString(transitions[row]));
    ret.append("\n");
  }
  return ret;
}

void PrintEval(const char* print_prefix, double loglik, int num_transitions) {
  printf("%-13s log-likelihood %f perplexity %f\n", print_prefix, loglik,
         Perplexity(loglik, num_transitions));
}

// Not used currently.
/*
void PrintEval(const char* print_prefix, double loglik, int num_transitions,
               const SparseMatrix& transition_probs) {
  printf("%-13s log-likelihood %f perplexity %f transitions:\n%s", print_prefix,
         loglik, Perplexity(loglik, num_transitions),
         ToString(transition_probs).c_str());
}
*/

std::string ToTSVString(const MetricsMap& metrics) {
  char buffer[512];
  sprintf(buffer, "%g\t%g\t%g\t%g", metrics.at("loglik"),
          metrics.at("perplexity"), metrics.at("accuracy"),
          metrics.at("num_params"));
  return buffer;
}

// NOLINTEND(runtime/printf)
