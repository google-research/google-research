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

#include <stdio.h>
#include <time.h>

#include <gflags/gflags.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <unordered_set>

// NOLINTBEGIN(build/include)
#include "common.h"
#include "io.h"
#include "gradient.h"
// NOLINTEND(build/include)

DEFINE_string(
    dataset, "you-forgot-to-set-dataset-flag",
    "Data set to use. Must be one of brightkite|lastfm|wiki|text|reuters|toy");
DEFINE_string(brightkite, "data/loc-brightkite_totalCheckins.txt",
              "File with Brightkite check-ins.");
DEFINE_string(
    lastfm,
    "data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv",
    "File with Last.fm 1K data.");
DEFINE_string(text_file, "data/romeo_juliet.txt", "Text file to read");
DEFINE_int32(lastfm_max_users, 100,
             "Read at most this many users from gigantic Last.fm data");
DEFINE_string(wiki, "data/wiki_paths.txt", "File with Wikispeedia page paths.");
DEFINE_int32(min_location_count, 50,
             "Replace all locations with a generic unknown location whose "
             "frequency is below this threshold");
DEFINE_int32(max_outer_iter, 1,
             "Maximum number of iterations in alternating optimization");
DEFINE_int32(grad_max_weight_iter, 30,
             "Gradient ascent maximum number of iterations when optimizing "
             "weight vector");
DEFINE_int32(grad_max_transitions_iter, 30,
             "Gradient ascent maximum number of iterations when optimizing "
             "the transition probablilities matrix");
DEFINE_double(grad_tolerance, 1e-4,
              "Gradient ascent is stopped if neither of the parameters changed "
              "by more than this amount.");
DEFINE_double(grad_learning_rate, 0.01, "Gradient ascent learning rate");
DEFINE_string(grad_method, "balance",
              "Method to update and contsrain parameters to the unit simplex "
              "in each step of the gradient ascent. Must be one of "
              "project,normalize,greedy,balance");
DEFINE_bool(bruteforce, false,
            "Whether to run brute-force search for the best 2 parameter weight "
            "combination with the observed transition matrix");
DEFINE_bool(lamp, true, "Whether to fit LAMP models up to --max_lamp_order");
DEFINE_int32(max_lamp_order, 7, "Max LAMP order");
DEFINE_int32(min_lamp_order, 1, "Min LAMP order");
DEFINE_bool(
    ngram, true,
    "Whether to fit higher order Markov chains up to order --max_ngram_order");
DEFINE_int32(max_ngram_order, 5, "Max N-gram order");
// TODO(stamas): Implement log level 0.
DEFINE_int32(debug_level, 1, "0: no debug 1: some 2: all");
DEFINE_string(max_train_time, "",
              "If not empty: max timestamp of visits used for training, newer "
              "data is used for testing. Overrides incompatible num_folds");
DEFINE_int32(num_folds, 2,
             "Number of cross validation folds, if <= 1 then the entire data "
             "set is reused for training and testing");
DEFINE_int32(random_seed, 20151006, "Random seed used in cross-validation");
DEFINE_string(
    plot_file, "../data/performance.tsv",
    "Name of file to write various permonace metrics into; used for plotting");

double SecondsElapsed(clock_t start) {
  clock_t now = clock();
  return static_cast<double>(now - start) / CLOCKS_PER_SEC;
}

void PrintEvals(const TrailsSlice& train_slice, const TrailsSlice& test_slice,
                double sum_freq, const WeightVector& item_frequencies,
                const SparseMatrix& transition_probs,
                const WeightVector& weights, EvalMap* eval_map) {
  const int order = weights.size();
  if (false && order != 4 && order != 5) {
    return;
  }
  Model model = {sum_freq, item_frequencies, transition_probs, weights};
  PrintAndAddEval("Initial", order, 0, true, train_slice, model, eval_map);
  PrintAndAddEval("Initial", order, 0, true, test_slice, model, eval_map);

  std::map<std::string, OptimMethod> name_to_method = {{"project", PROJECT},
                                                       {"normalize", NORMALIZE},
                                                       {"greedy", GREEDY},
                                                       {"balance", BALANCE}};
  const auto& it = name_to_method.find(FLAGS_grad_method);
  if (it == name_to_method.end()) {
    die("Unknown optimization method", FLAGS_grad_method);
  }
  OptimOptions options = {FLAGS_max_outer_iter,
                          FLAGS_grad_max_weight_iter,
                          FLAGS_grad_max_transitions_iter,
                          FLAGS_grad_tolerance,
                          FLAGS_grad_learning_rate,
                          it->second,
                          FLAGS_debug_level};
  // Optimize weights only.
  //  model.weights = OptimizeWeights(trails, model, options, 0);
  //  PrintEval(trails, model, "Optimized weights");
  clock_t start = clock();
  Optimize(train_slice, model, options, test_slice, "LAMP", eval_map);
  printf("Optimized LAMP in %g secs\n", SecondsElapsed(start));
}

void PrintEvals(const TrailsSlice& train_slice, const TrailsSlice& test_slice,
                double sum_freq, const WeightVector& item_frequencies,
                const SparseMatrix& transition_probs, int order,
                EvalMap* eval_map) {
  WeightVector weights(order);
  double w = 1;
  for (int i = 0; i < order; i++) {
    weights[i] = w;
    w *= 0.8;
  }
  Normalize(&weights);
  PrintEvals(train_slice, test_slice, sum_freq, item_frequencies,
             transition_probs, weights, eval_map);
}

void GridSearch(const TrailsSlice& train_slice, const TrailsSlice& test_slice,
                double sum_freq, const WeightVector& item_frequencies,
                const SparseMatrix& transitions) {
  clock_t start = clock();
  double best_log_lik = std::numeric_limits<double>::lowest();
  WeightVector best_weights;
  for (double w1 = 0.025; w1 < 1; w1 += 0.025) {
    Model model = {sum_freq, item_frequencies, transitions, {w1, 1.0 - w1}};
    Eval eval = PrintEval(train_slice, model, "Grid search");
    if (best_log_lik < eval.loglik()) {
      best_log_lik = eval.loglik();
      best_weights = model.weights;
    }
  }
  printf("Ran grid search in %g secs\n", SecondsElapsed(start));
  PrintEval(train_slice,
            {sum_freq, item_frequencies, transitions, best_weights},
            "Grid best");
  PrintEval(test_slice, {sum_freq, item_frequencies, transitions, best_weights},
            "Grid best");
}

void PrintAndAddEval(const TrailsSlice& trails, const NGramModel& ngram_model,
                     EvalMap* eval_map) {
  double loglik = LogLikelihood(trails, ngram_model);
  double perplexity = Perplexity(loglik, NumTransitions(trails));
  int num_params = 0;
  for (const auto& it : ngram_model.counts) {
    num_params += it.second.size();
  }
  printf("NGram order=%d %s log-likelihood %f perplexity %f num params %d\n",
         ngram_model.order, trails.is_test_ ? "test" : "train", loglik,
         perplexity, num_params);
  Eval eval;
  eval.metrics = {{"loglik", loglik},
                  {"perplexity", perplexity},
                  {"accuracy", 0.0},  // TODO(stamas): Implement me!
                  {"num_params", num_params}};
  (*eval_map)[{"NGram", ngram_model.order, 0, true, trails.is_test_}].push_back(
      eval);
}

void PrintAndAddEval(const TrailsSlice& trails, const KneyserNeyModel& kn_model,
                     EvalMap* eval_map) {
  double discount = 0.25;
  double loglik = LogLikelihood(trails, kn_model, discount);
  double perplexity = Perplexity(loglik, NumTransitions(trails));
  int num_params =
      kn_model.num_antecedents.size() + kn_model.middle_counts.size();
  for (const auto& it : kn_model.counts) {
    num_params += it.second.size();
  }
  printf(
      "KneyserNey order=%d discount=%g %s log-likelihood %f perplexity %f num "
      "params %d\n",
      kn_model.order, discount, trails.is_test_ ? "test" : "train", loglik,
      perplexity, num_params);
  Eval eval;
  eval.metrics = {{"loglik", loglik},
                  {"perplexity", perplexity},
                  {"accuracy", 0.0},  // TODO(stamas): Implement me!
                  {"num_params", num_params}};
  (*eval_map)[{"KneyserNey", kn_model.order, 0, true, trails.is_test_}]
      .push_back(eval);
}

void FitNGramModels(const TrailsSlice& train_slice,
                    const TrailsSlice& test_slice, int num_locations,
                    int max_order, EvalMap* eval_map) {
  for (int order = 1; order <= max_order; order++) {
    clock_t start = clock();
    NGramModel model = CountNGrams(train_slice, order);
    model.num_locations = num_locations;
    printf("Computed NGram matrix order %d in %g secs\n", order,
           SecondsElapsed(start));
    PrintAndAddEval(train_slice, model, eval_map);
    PrintAndAddEval(test_slice, model, eval_map);
    start = clock();
    KneyserNeyModel kn_model = CountKneyserNey(train_slice, order);
    kn_model.num_locations = num_locations;
    printf("Computed Kneyser-Ney model order %d in %g secs\n", order,
           SecondsElapsed(start));
    PrintAndAddEval(train_slice, kn_model, eval_map);
    PrintAndAddEval(test_slice, kn_model, eval_map);
  }
}

double stdev(double sum, double sum_sq, int n) {
  if (n > 1) {
    double variance = std::max((sum_sq - sum * (sum / n)) / (n - 1), 0.0);
    return sqrt(variance);
  } else {
    return 0;
  }
}

void SummarizeEvals(const std::vector<Eval>& evals, const std::string& metric,
                    double* mean, double* the_stdev) {
  double sum = 0, sum_sq = 0;
  for (const Eval& eval : evals) {
    double value = eval.metrics.at(metric);
    sum += value;
    sum_sq += value * value;
  }
  die_if(evals.empty(), "Empty eval vector");
  int expected_eval_size = FLAGS_max_train_time.empty() ? FLAGS_num_folds : 1;
  die_if(static_cast<int>(evals.size()) != expected_eval_size,
         "Unexpected eval vector size");
  *mean = sum / evals.size();
  *the_stdev = stdev(sum, sum_sq, evals.size());
}

void SummarizeEvals(const std::vector<Eval>& evals, Eval* mean_eval,
                    Eval* stdev_eval) {
  for (const std::string metric :
       {"loglik", "perplexity", "accuracy", "num_params"}) {
    SummarizeEvals(evals, metric, &mean_eval->metrics[metric],
                   &stdev_eval->metrics[metric]);
  }
  mean_eval->weights.clear();
  stdev_eval->weights.clear();
  if (evals.empty()) {
    return;
  }
  int num_weights = evals.front().weights.size();
  mean_eval->weights.resize(num_weights);
  stdev_eval->weights.resize(num_weights);
  WeightVector sum_weights(num_weights), sum_sq_weights(num_weights);
  for (const Eval& eval : evals) {
    die_if(static_cast<int>(eval.weights.size()) != num_weights,
           "Unexpected number of weights in SummarizeEvals");
    for (int i = 0; i < num_weights; i++) {
      sum_weights[i] += eval.weights[i];
      sum_sq_weights[i] += eval.weights[i] * eval.weights[i];
    }
  }
  for (int i = 0; i < num_weights; i++) {
    mean_eval->weights[i] = sum_weights[i] / evals.size();
    stdev_eval->weights[i] =
        stdev(sum_weights[i], sum_sq_weights[i], evals.size());
  }
}

void WritePlotData(const EvalMap& eval_map, FILE* plot_file) {
  fprintf(
      plot_file,
      "Algorithm\tOrder\tIter\tIterType\tTrainOrTest\t"
      "LogLikelihood\tPerplexity\tAccuracy\tNumParams\t"
      "LogLikelihoodStDev\tPerplexityStDev\tAccuracyStdDev\tNumParamsStDev\t"
      "Weights\tWeightsStDev\n");
  for (const auto& it : eval_map) {
    Eval mean_eval, stdev_eval;
    SummarizeEvals(it.second, &mean_eval, &stdev_eval);
    const Algorithm& algorithm = it.first;
    const char* iter_type = "final";
    if (!algorithm.is_final_iter) {  // first is used for weight only LAMP plots
      iter_type = algorithm.iter == 0 ? "first" : "intermediate";
    }
    double fractional_iter =
        (algorithm.iter + 1) / 2.0;  // weight iters become *.5 iter
    fprintf(plot_file, "%s\t%d\t%g\t%s\t%s\t%s\t%s\t%s\t%s\n",
            algorithm.name.c_str(), algorithm.order, fractional_iter, iter_type,
            algorithm.is_test ? "test" : "train",
            ToTSVString(mean_eval.metrics).c_str(),
            ToTSVString(stdev_eval.metrics).c_str(),
            ToString(mean_eval.weights).c_str(),
            ToString(stdev_eval.weights).c_str());
  }
}

void RunMethods(const TrailsSlice& train_slice, const TrailsSlice& test_slice,
                int num_locations, int cv_fold, EvalMap* eval_map) {
  std::unordered_set<int> train_loc, test_loc;
  for (const Trail& trail : train_slice) {
    for (int loc : trail) {
      train_loc.insert(loc);
    }
  }
  for (const Trail& trail : test_slice) {
    for (int loc : trail) {
      test_loc.insert(loc);
    }
  }
  int num_both = 0;
  for (int loc : test_loc) {
    if (train_loc.count(loc) > 0) {
      num_both++;
    }
  }
  if (cv_fold >= 0) {
    printf("Cross validation fold %d ", cv_fold);
  } else {
    printf("Time based split ");
  }
  printf("common location %d train only %d test only %d\n", num_both,
         static_cast<int>(train_loc.size()) - num_both,
         static_cast<int>(test_loc.size()) - num_both);

  clock_t start = clock();
  WeightVector item_frequencies;
  SparseMatrix empirical_transitions = ComputeEmpiricalTransitionMatrix(
      train_slice, num_locations, &item_frequencies);
  double sum_freq = Sum(item_frequencies);
  printf(
      "Computed empiricial transition matrix and item frequencies in %g secs\n",
      SecondsElapsed(start));
  if (FLAGS_dataset == "toy") {  // Overwrite with true matrix.
    empirical_transitions.clear();
    empirical_transitions.resize(3);
    empirical_transitions[0][0] = 0.5;
    empirical_transitions[0][1] = 0.5;
    empirical_transitions[1][2] = 1.0;
    empirical_transitions[2][0] = 1.0;
  }
  int num_entries = 0;
  for (const SparseVector& row : empirical_transitions) {
    num_entries += row.size();
  }
  printf("Empricial transition matrix has %d non-zero entries\n", num_entries);

  if (true || FLAGS_lamp) {
    PrintEvals(train_slice, test_slice, sum_freq, item_frequencies,
               empirical_transitions, 1, eval_map);
  }
  // Perform brute force grid search for the best 2 weights.
  if (FLAGS_bruteforce) {
    GridSearch(train_slice, test_slice, sum_freq, item_frequencies,
               empirical_transitions);
  }
  if (FLAGS_lamp) {
    int max_order = FLAGS_dataset == "toy" ? 2 : FLAGS_max_lamp_order;
    for (int order = std::max(FLAGS_min_lamp_order, 2); order <= max_order;
         order++) {
      PrintEvals(train_slice, test_slice, sum_freq, item_frequencies,
                 empirical_transitions, order, eval_map);
    }
    // Initial weight vectors were: {0.6, 0.4}, {0.6, 0.35, 0.05}, {0.6, 0.35,
    // 0.03, 0.02}, {0.6, 0.33, 0.03, 0.02, 0.02}, {0.6, 0.32, 0.03, 0.02, 0.02,
    // 0.01}
  }
  if (FLAGS_ngram) {
    int max_order = FLAGS_dataset == "toy" ? 2 : FLAGS_max_ngram_order;
    FitNGramModels(train_slice, test_slice, num_locations, max_order, eval_map);
  }
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Disable buffering to make log messages appear immediately.
  setbuf(stdout, NULL);

  // Die early if we can't write the results anyway.
  FILE* plot_file = fopen(FLAGS_plot_file.c_str(), "w");
  die_if(!plot_file, "Can't open", FLAGS_plot_file);
  EvalMap eval_map;

  // Read or create data.
  Trails trails, test_trails;  // latter only populated if splitting test and
                               // train by time.
  int num_locations = -1;
  bool split_by_time = !FLAGS_max_train_time.empty();
  if (FLAGS_dataset == "toy") {
    // Maithra's example for a LAMP that can't be expresses as a 1st order
    // Markov chain. Node indices are {0, 1, 2} instead of {1, 2, 3}.
    // Optimal weights are [0.5, 0.5].
    trails = {{0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1},
              {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 1, 2}};
    num_locations = 3;
  } else {
    clock_t start = clock();
    SplitStringTrails string_trails;
    if (FLAGS_dataset == "brightkite") {
      string_trails = ReadBrightkite(FLAGS_brightkite, FLAGS_max_train_time);
    } else if (FLAGS_dataset == "lastfm") {
      string_trails = ReadLastfm(FLAGS_lastfm, FLAGS_lastfm_max_users,
                                 FLAGS_max_train_time);
    } else if (FLAGS_dataset == "wiki") {
      die_if(split_by_time, "Wiki data can't be split by time");
      string_trails = ReadWiki(FLAGS_wiki);
    } else if (FLAGS_dataset == "text") {
      // TODO(stamas): split into sentences
      split_by_time = true;  // Fake mode
      FLAGS_max_train_time = "dummy";
      string_trails = ReadText(FLAGS_text_file);
    } else if (FLAGS_dataset == "reuters") {
      split_by_time = true;  // Fake mode
      FLAGS_max_train_time = "dummy";
      string_trails = ReadReuters();
    } else {
      die("Unknown dataset type", FLAGS_dataset);
    }
    ToTrails(string_trails, FLAGS_min_location_count, split_by_time, &trails,
             &test_trails, &num_locations);
    printf("Read %zu trails with %d transitions total in %g secs\n",
           string_trails.size(), NumTransitions(string_trails),
           SecondsElapsed(start));
  }

  if (split_by_time) {
    int fake_fold = -1;
    int num_train_transitions = NumTransitions(trails);
    int num_test_transitions = NumTransitions(test_trails);
    double denom = 0.01 * (num_train_transitions + num_test_transitions);
    printf(
        "Using %zu time based train trails with %d transitions total, %.2f%% "
        "of all, %.2f transitions/trail on average\n",
        trails.size(), num_train_transitions, num_train_transitions / denom,
        num_train_transitions / static_cast<double>(trails.size()));
    printf(
        "Using %zu time based test trails with %d transitions total, %.2f%% of "
        "all, %.2f transitions/trail on average\n",
        test_trails.size(), num_test_transitions, num_test_transitions / denom,
        num_test_transitions / static_cast<double>(test_trails.size()));
    RunMethods(TrailsSlice(trails, false), TrailsSlice(test_trails, true),
               num_locations, fake_fold, &eval_map);
  } else {  // cross validate
    std::mt19937 rng;
    rng.seed(FLAGS_random_seed);
    std::shuffle(trails.begin(), trails.end(), rng);
    die_if(FLAGS_num_folds < 1, "Number of folds must be at least one");
    die_if(static_cast<int>(trails.size()) < FLAGS_num_folds,
           "Too few trails for requested number of folds");
    int test_fold_size = trails.size() / FLAGS_num_folds;
    int train_fold_size =
        FLAGS_num_folds > 1 ? trails.size() - test_fold_size : trails.size();
    for (int fold = 0; fold < FLAGS_num_folds; fold++) {
      int test_begin = fold * test_fold_size;
      int test_end = (fold + 1) * test_fold_size;
      int train_begin = test_end % trails.size();
      int train_end = train_begin + train_fold_size;
      if (FLAGS_num_folds > 1) {
        train_end %= trails.size();
      }
      printf(
          "Cross validation fold %d test fold size %d train fold size %d "
          "test begin %d test end %d train begin %d train end %d\n",
          fold, test_fold_size, train_fold_size, test_begin, test_end,
          train_begin, train_end);
      TrailsSlice test_slice(trails, test_begin, test_end, true),
          train_slice(trails, train_begin, train_end, false);
      RunMethods(train_slice, test_slice, num_locations, fold, &eval_map);
    }
  }

  WritePlotData(eval_map, plot_file);
  die_if(fclose(plot_file) != 0, "Can't close", FLAGS_plot_file);
  return 0;
}
