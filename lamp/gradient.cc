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

#include "gradient.h"  // NOLINT(build/include)

#include <algorithm>
#include <cmath>

#include "common.h"  // NOLINT(build/include)

// NOLINTBEGIN(runtime/printf)

void AddTransitionGradient(const Trail& trail, const Model& model,
                           SparseMatrix* gradient, SparseMatrix* hessian_diag,
                           double* current_log_likelihood) {
  WeightVector prob_coeffs(model.weights.size());  // comes from smoothing
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    double prob = 0;
    int col = trail[i + 1];
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      prob += model.weights[j] * model.GetProb(row, col, &prob_coeffs[j]);
    }
    check_prob(prob, "transition probability in transition gradient");
    // Partial derivatives of log(prob) wrt prob:row -> col inside GetProb(row,
    // col)
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      double tmp = model.weights[j] * prob_coeffs[j] / prob;
      (*gradient)[row][col] += tmp;
      (*hessian_diag)[row][col] += tmp * tmp;
    }
    *current_log_likelihood += log(prob);
  }
}

SparseMatrix TransitionGradient(const TrailsSlice& trails, const Model& model,
                                SparseMatrix* hessian_diag,
                                double* current_log_likelihood) {
  SparseMatrix gradient(model.transition_probs.size());
  hessian_diag->clear();
  hessian_diag->resize(model.transition_probs.size());
  *current_log_likelihood = 0;
  for (const Trail& trail : trails) {
    AddTransitionGradient(trail, model, &gradient, hessian_diag,
                          current_log_likelihood);
  }
  // To ease debugging return average gradient
  Scale(1.0 / NumTransitions(trails), &gradient);
  // -1 adds missing sign we omitted from numerous updates
  Scale(-1.0 / NumTransitions(trails), hessian_diag);
  return gradient;
}

// In optimum all gradients of 0 < ... < 1 transitions are equal by the KKT
// condition. Updates transition probabilities toward that goal while keeping
// their sum unchanged. Input 'hessian' contains the diagonal elements of the
// Hessian matrix.
struct Breakpoint {
  double lambda;    // gradient value attained at the breakpoint
  bool lambda_max;  // If true breakpoint corresponds to highest gradient for
                    // this column attained if variable = 0. If false then
                    // lambda is the gradient attained at variable = 1.
  int column;       // variable index

  bool operator<(const Breakpoint& rhs) const { return lambda < rhs.lambda; }
};

void BalanceTransitions(SparseVector& gradient, const SparseVector& hessian,
                        double max_allowed_change, char step_type,
                        int outer_iter, int iter, int row, bool debug_log,
                        SparseVector* transition_probs, double* max_delta,
                        double* estimated_log_lik_change) {
  // Ignore 0 gradients (as then the hessian, which we divide by, is 0 too).
  // This pathological case can only happen if the corresponding weight is 0.
  *estimated_log_lik_change = 0;
  for (auto it = gradient.begin(); it != gradient.end(); /* no increment */) {
    if (std::fabs(it->second) < 1e-12) {
      gradient.erase(it++);
    } else {
      ++it;
    }
  }
  if (gradient.empty()) {
    if (debug_log) {
      printf("%c-step %d/%d row=%d empty gradients, nothing to do\ncurr\t%s\n",
             step_type, outer_iter, iter, row,
             ToString(*transition_probs).c_str());
    }
    return;
  }
  SparseVector old_transition_probs = *transition_probs;
  // If a transition probability never occurred in the training data, set it to
  // 0.
  double sum_transition_prob_zeroed = 0;
  for (auto it = transition_probs->cbegin(); it != transition_probs->cend();
       /* no increment */) {
    if (gradient.find(it->first) == gradient.end()) {
      sum_transition_prob_zeroed += it->second;
      transition_probs->erase(it++);
    } else {
      ++it;
    }
  }
  // KKT condition.
  std::vector<Breakpoint> breakpoints;
  breakpoints.reserve(2 * gradient.size());
  double sum_disallowed_down_adjustments = 0;
  for (const auto& column_and_value : gradient) {
    int column = column_and_value.first;
    double grad = column_and_value.second;
    double hessian_value = hessian.at(column_and_value.first);
    double curr_value = GetOrZero(*transition_probs, column);
    sum_disallowed_down_adjustments +=
        std::max(curr_value - max_allowed_change, 0.0);
    // Assumes negative hessian
    breakpoints.push_back(
        {grad - std::min(curr_value, max_allowed_change) * hessian_value, true,
         column});  // lambda_max
    breakpoints.push_back(
        {grad + std::min(1.0 - curr_value, max_allowed_change) * hessian_value,
         false, column});  // lambda_min
  }
  std::sort(breakpoints.begin(), breakpoints.end());
  // Call a variable active if it's 0 < ... < 1.
  double sum_delta_inactive = -1.0 + sum_disallowed_down_adjustments;
  // Sum of delta of vars set to either 0 or 1.
  // Initially all vars are set to 0, hence the
  // value of this is (-1) * sum of transition
  // probabilities, which is -1.
  double sum_active_g_over_h = 0, sum_active_one_over_h = 0;
  // Common gradient of active variables
  double lambda = -1;  // assumes positive gradients.
  for (int i = breakpoints.size() - 1; i >= 1;
       i--) {  // last breakpoint sets all vars to 1.
    const Breakpoint& breakpoint = breakpoints[i];
    double curr_value = GetOrZero(*transition_probs, breakpoint.column);
    double grad = gradient.at(breakpoint.column);
    double hessian_value = hessian.at(breakpoint.column);
    if (breakpoint.lambda_max) {  // variable indexed by column became active,
                                  // i.e. > 0.
      sum_delta_inactive +=
          std::min(curr_value, max_allowed_change);  // minus (-curr_value)
      sum_active_g_over_h += grad / hessian_value;
      sum_active_one_over_h += 1.0 / hessian_value;
    } else {  // variable indexed by column became inactive again, its value
              // reached 1.
      // This shouldn't really happen for multiple variables. Maybe exit early??
      // We could also exit at lambda < 0, or not add them at all?
      sum_delta_inactive += std::min(1.0 - curr_value, max_allowed_change);
      sum_active_g_over_h -= grad / hessian_value;
      sum_active_one_over_h -= 1.0 / hessian_value;
    }
    lambda = (/*sum_transition_prob_zeroed + */ sum_active_g_over_h -
              sum_delta_inactive) /
             sum_active_one_over_h;
    if (debug_log) {
      printf(
          "breakpoint i %d col %d grad %g hessian %g curr value %g lambda %s "
          "%g target lambda %g sum_delta_inactive %g sum_active_g_over_h %g "
          "sum_active_one_over_h %g nom_shift %g\n",
          i, breakpoint.column, grad, hessian_value, curr_value,
          breakpoint.lambda_max ? "max" : "min", breakpoint.lambda, lambda,
          sum_delta_inactive, sum_active_g_over_h, sum_active_one_over_h,
          sum_transition_prob_zeroed - sum_delta_inactive);
    }
    // comparison tolerance
    const double eps = 1e-14;
    // Last breakpoint OR the curr and prev breakpoints are not equal.
    bool acceptable_breakpoint =
        i == 1 || (breakpoints[i - 1].lambda) + 4 * eps <= breakpoint.lambda;
    if (acceptable_breakpoint && breakpoints[i - 1].lambda <= (lambda + eps) &&
        lambda <= (breakpoint.lambda + eps)) {
      // Lambda is feasible with current set of active nodes.
      break;
    }
  }
  if (debug_log) {
    printf("%c-step %d/%d row=%d lambda=%g\ncurr\t%s\ngrad\t%s\nhessian\t%s\n",
           step_type, outer_iter, iter, row, lambda,
           ToString(*transition_probs).c_str(), ToString(gradient).c_str(),
           ToString(hessian).c_str());
  }
  SparseVector adjustments;
  for (const auto& column_and_value : gradient) {
    int column = column_and_value.first;
    double old_value = (*transition_probs)[column];
    double adjust = (lambda - column_and_value.second) / hessian.at(column);
    adjust =
        std::min(std::max(adjust, -max_allowed_change), max_allowed_change);
    // Adjust and clip to [0,1].
    double new_value = std::max(std::min(old_value + adjust, 1.0), 0.0);
    if (new_value < 1e-10) {
      // Drop (near) zero values.
      transition_probs->erase(column);
    } else {
      (*transition_probs)[column] = new_value;
    }
    adjust = new_value - old_value;
    if (std::fabs(adjust) >= max_allowed_change + 1e-4) {
      printf("row %d col %d old %g -> new %g\n", row, column, old_value,
             new_value);
      die("Stepped outside trust region in BalanceTransitions");
    }
    *max_delta = std::max(*max_delta, std::fabs(adjust));
    if (debug_log) {
      adjustments[column] = adjust;
    }
    *estimated_log_lik_change += column_and_value.second * adjust +
                                 0.5 * hessian.at(column) * adjust * adjust;
  }
  double sum = Sum(*transition_probs);
  if (debug_log) {
    printf(
        "%c-step %d/%d row=%d sum of new probs=%g estimated log-lik "
        "change=%g\nadjustments\t%s\n",
        step_type, outer_iter, iter, row, sum, *estimated_log_lik_change,
        ToString(adjustments).c_str());
  }
  if (std::fabs(sum - 1.0) > 1e-4) {
    printf("%c-step %d/%d row=%d lambda=%g\nnew\t%s\ngrad\t%s\nhessian\t%s\n",
           step_type, outer_iter, iter, row, lambda,
           ToString(*transition_probs).c_str(), ToString(gradient).c_str(),
           ToString(hessian).c_str());
    printf("%c-step %d/%d row=%d sum of new probs=%g\nadjustments\t%s\n",
           step_type, outer_iter, iter, row, sum,
           ToString(adjustments).c_str());
    die("Highly non-unit transition probability sum in BalanceTransitions");
  }
  // Due to numerical errors, sum can become > 1+1e-6
  Scale(1.0 / sum, transition_probs);
}

SparseMatrix OptimizeTransitions(const TrailsSlice& trails,
                                 const Model& input_model,
                                 const OptimOptions& options, int outer_iter) {
  die_if(options.method != BALANCE,
         "Unimplemented optimization method in OptimizeTransitions");
  Model model = input_model;
  SparseMatrix hessian_diag;
  for (int i = 0; i < options.max_transitions_iter; i++) {
    double curr_log_likelihood;
    SparseMatrix gradient =
        TransitionGradient(trails, model, &hessian_diag, &curr_log_likelihood);
    char msg_prefix[128];
    sprintf(msg_prefix, "T-step %d/%d begin", outer_iter, i);
    PrintEval(msg_prefix, curr_log_likelihood, NumTransitions(trails));
    if (options.debug_level >= 2) {
      printf("transitions:\n%s", ToString(model.transition_probs).c_str());
    }
    double max_delta = 0;
    for (size_t row = 0; row < gradient.size(); row++) {
      double max_allowed_change = 1.1;  // should have no effect
      double unused_estimated_log_lik_change;
      BalanceTransitions(gradient[row], hessian_diag[row], max_allowed_change,
                         'T', outer_iter, i, row, (options.debug_level > 1),
                         &model.transition_probs[row], &max_delta,
                         &unused_estimated_log_lik_change);
    }
    printf("T-step %d/%d max transition prob change=%g\n", outer_iter, i,
           max_delta);
    if (max_delta < options.tolerance) {
      break;
    }
  }
  return model.transition_probs;
}

// Transition gradients for 1 row.
void NewTransitionGradient(const PostingList& posting_list, const Model& model,
                           int num_transitions, SparseVector* gradient,
                           SparseVector* hessian_diag,
                           double* current_log_likelihood) {
  gradient->clear();
  hessian_diag->clear();
  *current_log_likelihood = 0;
  WeightVector prob_coeffs(model.weights.size());  // comes from smoothing
  for (const TrailPositions& trail_positions : posting_list) {
    const Trail& trail = *trail_positions.trail;
    int prev_position = (-2) * static_cast<int>(model.weights.size());
    for (int position : trail_positions.positions) {
      for (size_t shift = 0;
           shift < model.weights.size() && position + shift + 1 < trail.size();
           shift++) {
        int col = trail[position + shift + 1];
        double prob = 0, nominator = 0;
        for (size_t j = 0; j < model.weights.size(); j++) {
          int row = Get(trail, position + shift, j);
          double prob_coeff;
          prob += model.weights[j] * model.GetProb(row, col, &prob_coeff);
          if (j == shift || (position == 0 && shift < j)) {
            if (row != trail[position]) {
              die("unexpected trail item in new transition gradient");
            }
            nominator += model.weights[j] * prob_coeff;
          }
        }
        check_prob(prob, "transition probability in new transition gradient");
        // Partial derivatives of log(prob) wrt prob:row -> col inside
        // GetProb(row, col)
        double tmp = nominator / prob;
        (*gradient)[col] += tmp;
        (*hessian_diag)[col] -= tmp * tmp;
        // Not overlapping
        if (position + static_cast<int>(shift) + 1 >
            prev_position + static_cast<int>(model.weights.size())) {
          *current_log_likelihood += log(prob);
        }
      }
      prev_position = position;
    }
  }
  // To ease debugging return average gradient
  Scale(1.0 / num_transitions, gradient);
  Scale(1.0 / num_transitions, hessian_diag);
}

SparseMatrix NewOptimizeTransitions(const TrailsSlice& trails,
                                    const PositionIndex& position_index,
                                    const Model& input_model,
                                    const OptimOptions& options,
                                    int outer_iter) {
  die_if(position_index.size() != input_model.transition_probs.size(),
         "Mismatched sizes in NewOptimizeTransitions");
  char msg_prefix[128];
  sprintf(msg_prefix, "T-step %d begin", outer_iter);
  PrintEval(trails, input_model, msg_prefix);

  const int num_transitions = NumTransitions(trails);
  Model model = input_model;
  SparseVector gradient, hessian_diag;
  double final_overall_max_delta = 0;
  int num_steps = -1;
  int num_major_undos = 0;
  // Maybe randomize row order?
  for (size_t row = 0; row < position_index.size(); row++) {
    SparseVector best_transition_probs;  // Initialized by first iteration.
    double max_delta = 1e10, best_partial_log_lik = -1e30,
           max_allowed_change = 1.0, estimated_log_lik_change = -1;
    bool after_balance = false;
    for (int i = 0;
         i < options.max_transitions_iter && max_delta >= options.tolerance;
         i++) {
      double partial_log_lik;
      NewTransitionGradient(position_index[row], model, num_transitions,
                            &gradient, &hessian_diag, &partial_log_lik);
      bool undo = (best_partial_log_lik > partial_log_lik);
      if (undo) {  // undo last change that decreased loglik
        model.transition_probs[row] = best_transition_probs;
      } else {
        best_transition_probs = model.transition_probs[row];
      }
      if (false) {
        // Trust region method. (surprisingly not much quicker and tiny bit less
        // accurate)
        double increase_ratio = (partial_log_lik - best_partial_log_lik) /
                                (num_transitions * estimated_log_lik_change);
        if (increase_ratio < 0.25 && after_balance) {
          max_allowed_change *= 0.25;
        } else if (increase_ratio > 0.75 && after_balance) {
          max_allowed_change = std::min(2 * max_allowed_change, 1.0);
        }
        if (undo) {
          // Make sure max_allowed_change has an effect when rerunning the
          // opt.
          max_allowed_change = std::min(max_allowed_change, max_delta * 0.25);
        }
      } else {
        // No proper trust adjustment, just shrink if log-lik is non-increasing.
        if (undo) {
          max_allowed_change = std::min(max_allowed_change, max_delta) * 0.1;
        }
      }
      // Always provide some progress while making number of lines independent
      // of number of items.
      // At least 1% worse.
      bool major_undo = (best_partial_log_lik * 1.01 > partial_log_lik);
      if (major_undo) {
        if (++num_major_undos > 20) {
          major_undo = false;  // silence warnings if too many
        }
      }
      if (++num_steps % position_index.size() == 0 || major_undo ||
          options.debug_level >= 2) {
        printf(
            "T-step %d/row %zu/%d curr partial log-likelihood=%f prev best "
            "partial "
            "log-likelihood=%f prev max delta=%g max_allowed_change=%g%s\n",
            outer_iter, row, i, partial_log_lik, best_partial_log_lik,
            max_delta, max_allowed_change, undo ? " MAJOR UNDO" : "");
        if (options.debug_level >= 2) {
          printf("transitions row: %s\n",
                 ToString(model.transition_probs[row]).c_str());
        }
      }
      if (undo || i == options.max_transitions_iter - 1) {
        after_balance = false;
        continue;
      }
      best_partial_log_lik = partial_log_lik;
      // BalanceTransitions does a max update on its max_delta param.
      max_delta = -1;
      BalanceTransitions(gradient, hessian_diag, max_allowed_change, 'T',
                         outer_iter, i, row, (options.debug_level > 1),
                         &model.transition_probs[row], &max_delta,
                         &estimated_log_lik_change);
      after_balance = true;
    }
    final_overall_max_delta = std::max(final_overall_max_delta, max_delta);
  }
  printf("T-step %d max final transition prob change=%g\n", outer_iter,
         final_overall_max_delta);

  return model.transition_probs;
}

// hessian_diag contains (-1) * diagonal of the Hessian matrix
void AddWeightGradient(const Trail& trail, const Model& model,
                       WeightVector* gradient, WeightVector* hessian_diag,
                       double* current_log_likelihood) {
  WeightVector out_transitions(model.weights.size());
  for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
    double prob = 0;
    int col = trail[i + 1];
    for (size_t j = 0; j < model.weights.size(); j++) {
      int row = Get(trail, i, j);
      out_transitions[j] = model.GetProb(row, col);
      prob += model.weights[j] * out_transitions[j];
    }
    check_prob(prob, "transition probability in weight gradient");
    // Partial derivatives of log(prob) wrt weights
    for (size_t j = 0; j < model.weights.size(); j++) {
      double tmp = out_transitions[j] / prob;
      (*gradient)[j] += tmp;
      (*hessian_diag)[j] += tmp * tmp;
    }
    *current_log_likelihood += log(prob);
  }
}

WeightVector WeightGradient(const TrailsSlice& trails, const Model& model,
                            WeightVector* hessian_diag,
                            double* current_log_likelihood) {
  WeightVector gradient(model.weights.size());
  hessian_diag->clear();
  hessian_diag->resize(model.weights.size());
  *current_log_likelihood = 0;
  for (const Trail& trail : trails) {
    AddWeightGradient(trail, model, &gradient, hessian_diag,
                      current_log_likelihood);
  }
  // To ease debugging return average gradient
  Scale(1.0 / NumTransitions(trails), &gradient);
  // -1 adds missing sign we omitted from numerous updates
  Scale(-1.0 / NumTransitions(trails), hessian_diag);
  return gradient;
}

// Finds the point of the unit simplex closest to weights.
// Implements Figure 1 of http://www.magicbroom.info/Papers/DuchiShSiCh08.pdf
// In the paper's notation v = weights argument above, z = 1.
void Project(WeightVector* weights) {
  // Sort descending
  WeightVector sorted_weights = *weights;
  std::sort(sorted_weights.rbegin(), sorted_weights.rend());
  int rho = -1;
  double sum_sorted_weights = 0;
  for (size_t j = 0; j < sorted_weights.size(); j++) {
    sum_sorted_weights += sorted_weights[j];
    if (sorted_weights[j] - (sum_sorted_weights - 1.0) / (j + 1) <= 0) {
      sum_sorted_weights -= sorted_weights[j];
      break;
    }
    rho = j;
  }
  double theta = (sum_sorted_weights - 1.0) / (rho + 1);
  for (double& weight : *weights) {
    weight = std::max(weight - theta, 0.0);
  }
}

// Increases the weight with largest gradient by at most learning_rate and
// descreases the smallest positive weights by the same amount to keep the
// sum of weights (=1) unchanged.
void GreedyWeightUpdate(const WeightVector& gradient, double learning_rate,
                        WeightVector* weights) {
  std::vector<std::pair<double, int>> sorted_gradient;
  for (size_t j = 0; j < gradient.size(); j++) {
    sorted_gradient.emplace_back(gradient[j], j);
  }
  std::sort(sorted_gradient.begin(), sorted_gradient.end());
  int max_grad_idx = sorted_gradient.back().second;
  double eps = std::min(learning_rate, 1.0 - (*weights)[max_grad_idx]);
  die_if(eps < 0, "Negative adjustment epsilon in GreedyWeightUpdate");
  (*weights)[max_grad_idx] += eps;
  for (size_t j = 0; j + 1 < gradient.size(); j++) {
    int idx = sorted_gradient[j].second;
    if ((*weights)[idx] >= eps) {
      (*weights)[idx] -= eps;
      break;
    } else {
      eps -= (*weights)[idx];
      (*weights)[idx] = 0;
    }
  }
}

SparseVector WeightToSparseVector(const WeightVector& v) {
  SparseVector ret;
  for (size_t i = 0; i < v.size(); i++) {
    ret[i] = v[i];
  }
  return ret;
}

WeightVector SparseToWeightVector(const SparseVector& v, int order) {
  WeightVector ret(order);
  for (const auto& index_and_value : v) {
    ret.at(index_and_value.first) = index_and_value.second;
  }
  return ret;
}

// In optimum all gradients of 0 < ... < 1 weights are equal by the KKT
// condition. Updates weights toward that goal while keeping their sum
// unchanged. Input 'hessian' contains the diagonal elements of the Hessian
// matrix.
void BalanceWeights(const WeightVector& gradient, const WeightVector& hessian,
                    int outer_iter, int iter, bool debug_log,
                    WeightVector* weights) {
  // Reuses SparseVector code for transitions.
  SparseVector gradient_as_sv = WeightToSparseVector(gradient);
  const SparseVector hessian_as_sv = WeightToSparseVector(hessian);
  SparseVector weights_as_sv = WeightToSparseVector(*weights);
  const int kDummyRowForDebugLogs = -1;
  double unused_max_delta = 0, unused_estimated_log_lik_change = -1;
  double max_allowed_change = 1.1;  // should have no effect
  BalanceTransitions(gradient_as_sv, hessian_as_sv, max_allowed_change, 'W',
                     outer_iter, iter, kDummyRowForDebugLogs, debug_log,
                     &weights_as_sv, &unused_max_delta,
                     &unused_estimated_log_lik_change);
  *weights = SparseToWeightVector(weights_as_sv, gradient.size());
}

WeightVector OptimizeWeights(const TrailsSlice& trails,
                             const Model& input_model,
                             const OptimOptions& options, int outer_iter) {
  const bool debug_log = options.debug_level >= 2;
  Model model = input_model;
  WeightVector sum_gradient_squared(model.weights.size());
  int num_transitions = NumTransitions(trails);

  for (int i = 0; i < options.max_weight_iter; i++) {
    WeightVector prev_weights = model.weights;
    double curr_log_likelihood;
    WeightVector hessian_diag;
    WeightVector gradient =
        WeightGradient(trails, model, &hessian_diag, &curr_log_likelihood);
    char msg_prefix[128];
    sprintf(msg_prefix, "W-step %d/%d begin", outer_iter, i);
    PrintEval(msg_prefix, curr_log_likelihood, num_transitions, model.weights);
    if (debug_log) {
      printf("W-step %d/%d gradient %s\n", outer_iter, i,
             ToString(gradient).c_str());
    }
    if (options.method == GREEDY) {
      GreedyWeightUpdate(gradient, options.learning_rate, &model.weights);
    } else if (options.method == BALANCE) {
      BalanceWeights(gradient, hessian_diag, outer_iter, i, debug_log,
                     &model.weights);
    } else {
      for (size_t j = 0; j < gradient.size(); j++) {
        // AdaGrad from
        // J. Duchi, E. Hazan, and Y. Singer.
        // Adaptive subgradient methods for online learning and stochastic
        // optimization. JMLR, 2010.
        // http://www.colt2010.org/papers/023Duchi.pdf
        sum_gradient_squared[j] += gradient[j] * gradient[j];
        double step_size =
            options.learning_rate / sqrt(1 + sum_gradient_squared[j]);
        model.weights[j] += step_size * gradient[j];
      }
      if (debug_log) {
        printf("W-step %d/%d unconstrained new weight %s sum %g\n", outer_iter,
               i, ToString(model.weights).c_str(), Sum(model.weights));
      }
      if (options.method == PROJECT) {
        Project(&model.weights);
      } else if (options.method == NORMALIZE) {
        Normalize(&model.weights);
      } else {
        die("Unknown optimization method in OptimizeWeights");
      }
    }
    double max_delta = MaxAbsDifference(prev_weights, model.weights);
    printf("W-step %d/%d max weight change %g\n", outer_iter, i, max_delta);
    if (max_delta < 1e-3) {  // options.tolerance) {
      break;
    }
  }
  return model.weights;
}

Model Optimize(const TrailsSlice& trails, const Model& initial_model,
               const OptimOptions& options, const TrailsSlice& test_trails,
               const char* alg_name, EvalMap* eval_map) {
  Model model = initial_model;
  PositionIndex position_index =
      BuildIndex(trails, model.transition_probs.size());
  int order = model.weights.size();
  char msg_prefix[128];
  for (int i = 0; i < options.max_outer_iter; i++) {
    model.weights = OptimizeWeights(trails, model, options, i);
    sprintf(msg_prefix, "Step %d optimized weights", i);
    PrintEval(trails, model, msg_prefix);
    PrintAndAddEval(alg_name, order, 2 * i, false, trails, model, eval_map);
    PrintAndAddEval(alg_name, order, 2 * i, false, test_trails, model,
                    eval_map);

    if (true) {
      model.transition_probs =
          NewOptimizeTransitions(trails, position_index, model, options, i);
    } else {
      model.transition_probs = OptimizeTransitions(trails, model, options, i);
    }
    sprintf(msg_prefix, "Step %d optimized transitions", i);
    PrintEval(trails, model, msg_prefix);
    bool is_final_iter = (i + 1 == options.max_outer_iter);
    PrintAndAddEval(alg_name, order, 2 * i + 1, is_final_iter, trails, model,
                    eval_map);
    PrintAndAddEval(alg_name, order, 2 * i + 1, is_final_iter, test_trails,
                    model, eval_map);
  }
  return model;
}

PositionIndex BuildIndex(const TrailsSlice& trails, int num_locations) {
  PositionIndex index(num_locations);
  for (const Trail& trail : trails) {
    for (int i = 0; i < static_cast<int>(trail.size()) - 1; i++) {
      int item = trail[i];
      if (index.at(item).empty() || index[item].back().trail != &trail) {
        index[item].push_back(TrailPositions(&trail));
      }
      index[item].back().positions.push_back(i);
    }
  }
  return index;
}

// NOLINTEND(runtime/printf)

