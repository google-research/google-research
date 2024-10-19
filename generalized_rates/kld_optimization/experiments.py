# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run comparisons for optimizing KLD s.t. error rate constraint.

Specifically, compare different methods for solving:
  min sum KLD(p, hat{p}_G) s.t. error_rate <= eps * unconstrained error_rate,
  where p is the overall proportion of positives, hat{p}_G is the predicted
  proportion of positives for group G, and eps is a multiplicative slack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from generalized_rates.kld_optimization import evaluation
from generalized_rates.kld_optimization import methods
import numpy as np
import tensorflow_constrained_optimization as tfco


flags.DEFINE_string("data_file", None, "Path to dataset.")
flags.DEFINE_integer("loops_con", 5000,
                     "No. of iterations for Lagrangian optimizer.")
flags.DEFINE_integer("loops_unc", 2500,
                     "No. of iterations for unconstrained methods.")
flags.DEFINE_float("epsilon", 1.1, "Constraint slack.")

FLAGS = flags.FLAGS


def print_results(test_set, models, probabilities, title, error_unc=None):
  """Prints and returns KLD and error rate on test set.

  Args:
    test_set: (x_test, y_test, z_test)
    models: list of tuples (weights, threshold)
    probabilities: list of floats, containing classifier probabilities
    title: string, method name to print
    error_unc: optional float, the unconstrained classifier's error rate

  Returns:
    KLD objective, error rate
  """
  x_test, y_test, z_test = test_set

  error = evaluation.expected_error_rate(x_test, y_test, models, probabilities)
  klds = evaluation.expected_group_klds(
      x_test, y_test, z_test, models, probabilities)

  if error_unc is None:
    print(title + ": %.3f (%.3f, %.3f)" % (sum(klds), error, 1.0))
  else:
    print(title + ": %.3f (%.3f, %.3f)" % (sum(klds), error, error / error_unc))
  return sum(klds), error


def run_experiment():
  """Run experiments comparing unconstrained and constrained methods."""
  # Range of hyper-parameters for unconstrained and constrained optimization.
  lr_range_unc = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
  lr_range_con = [0.001, 0.01, 0.1, 1.0]

  # Load dataset.
  with open(FLAGS.data_file, "rb") as f:
    train_set, vali_set, test_set = np.load(
        f, allow_pickle=True, fix_imports=True)
  x_vali, y_vali, z_vali = vali_set

  ##################################################
  # Unconstrained Error Optimization.
  print("Running unconstrained error optimization")

  models_unc = []
  param_objectives_unc = []

  # Find best learning rate.
  for lr_model in lr_range_unc:
    model = methods.error_rate_optimizer(
        train_set, learning_rate=lr_model, loops=FLAGS.loops_unc)
    error = evaluation.expected_error_rate(x_vali, y_vali, [model], [1.0])
    param_objectives_unc.append(error)
    models_unc.append(model)

  best_param_index_unc = np.argmin(param_objectives_unc)
  model_er = models_unc[best_param_index_unc]
  print()

  ##################################################
  # Post-shift for Demographic Parity.
  print("Running post-shift for demographic parity")

  # First train logistic regression model.
  models_log = []
  param_objectives_log = []

  # Find best learning rate.
  for lr_model in lr_range_unc:
    model = methods.logistic_regression(
        train_set, learning_rate=lr_model, loops=FLAGS.loops_unc)
    loss = evaluation.cross_entropy_loss(x_vali, y_vali, model[0], model[1])
    param_objectives_log.append(loss)
    models_log.append(model)

  best_param_index_log = np.argmin(param_objectives_log)
  logreg_model = models_log[best_param_index_log]

  # Post-shift logistic regression model for demographic parity.
  model_ps, train_set_ps, vali_set_ps, test_set_ps = methods.post_shift_dp(
      train_set, vali_set, test_set, logreg_model)
  print()

  ##################################################
  # Surrogate-based Lagrangian Optimizer for Convex Rate Metrics (Algorithm 2).
  print("Running constrained Lagrangian optimization (Algorithm 2)")

  # Set additive slack to unconstrained error * epsilon.
  x_train, y_train, _ = train_set
  error_unc_train = evaluation.expected_error_rate(
      x_train, y_train, [model_er], [1.0])
  additive_slack = error_unc_train * FLAGS.epsilon

  # Maintain list of models, objectives and violations for hyper-parameters.
  stochastic_models_list = []
  deterministic_models_list = []
  param_objectives_con = []
  param_violations_con = []

  # Find best learning rates for model parameters and Lagrange multipliers.
  for lr_model in lr_range_con:
    for lr_constraint in lr_range_con:
      stochastic_model, deterministic_model = (
          methods.lagrangian_optimizer_kld(
              train_set,
              learning_rate=lr_model,
              learning_rate_constraint=lr_constraint,
              loops=FLAGS.loops_con,
              additive_slack=additive_slack
          )
      )
      stochastic_models_list.append(stochastic_model)
      deterministic_models_list.append(deterministic_model)

      # Record objective and constraint violations for stochastic model.
      klds = evaluation.expected_group_klds(
          x_vali, y_vali, z_vali, stochastic_model[0], stochastic_model[1])
      param_objectives_con.append(sum(klds))

      error = evaluation.expected_error_rate(
          x_vali, y_vali, stochastic_model[0], stochastic_model[1])
      param_violations_con.append([error - additive_slack])

      print("Parameters (%.3f, %.3f): %.3f (%.3f)" % (
          lr_model, lr_constraint, param_objectives_con[-1],
          max(param_violations_con[-1])))

  # Best param.
  best_param_index_con = tfco.find_best_candidate_index(
      np.array(param_objectives_con), np.array(param_violations_con))

  stochastic_model_con = stochastic_models_list[best_param_index_con]
  deterministic_model_con = deterministic_models_list[best_param_index_con]
  print()

  # Print summary of performance on test set.
  results = {}
  results["UncError"] = print_results(test_set, [model_er], [1.0], "UncError")
  error_unc = results["UncError"][1]
  results["PostShift"] = print_results(
      test_set_ps, [model_ps], [1.0], "PostShift", error_unc)
  results["Stochastic"] = print_results(
      test_set, stochastic_model_con[0], stochastic_model_con[1],
      "Constrained (Stochastic)", error_unc)
  results["Deterministic"] = print_results(
      test_set, [deterministic_model_con], [1.0], "Constrained (Deterministic)",
      error_unc)
  print()

  # Print summary of performance on train set.
  results = {}
  results["UncError"] = print_results(train_set, [model_er], [1.0], "UncError")
  error_unc = results["UncError"][1]
  results["PostShift"] = print_results(
      train_set_ps, [model_ps], [1.0], "PostShift", error_unc)
  results["Stochastic"] = print_results(
      train_set, stochastic_model_con[0], stochastic_model_con[1],
      "Constrained (Stochastic)", error_unc)
  results["Deterministic"] = print_results(
      train_set, [deterministic_model_con], [1.0],
      "Constrained (Deterministic)", error_unc)
  print()

  # Print summary of performance on vali set.
  results = {}
  results["UncError"] = print_results(vali_set, [model_er], [1.0], "UncError")
  error_unc = results["UncError"][1]
  results["PostShift"] = print_results(
      vali_set_ps, [model_ps], [1.0], "PostShift", error_unc)
  results["Stochastic"] = print_results(
      vali_set, stochastic_model_con[0], stochastic_model_con[1],
      "Constrained (Stochastic)", error_unc)
  results["Deterministic"] = print_results(
      vali_set, [deterministic_model_con], [1.0], "Constrained (Deterministic)",
      error_unc)


def main(argv):
  del argv
  run_experiment()


if __name__ == "__main__":
  app.run(main)
