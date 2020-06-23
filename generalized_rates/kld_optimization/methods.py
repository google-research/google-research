# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Unconstrained and constrained methods F-measure optimization.

Specifically, compare different methods for solving:
  min sum KLD(p, hat{p}_G) s.t. error_rate <= additive_slack,
  where p is the overall proportion of positives and hat{p}_G is the predicted
  proportion of positives for group G.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from generalized_rates.kld_optimization import evaluation
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco


def error_rate_optimizer(train_set, learning_rate, loops):
  """Returns a model that optimizes the hinge loss."""
  x_train, y_train, _ = train_set
  dimension = x_train.shape[-1]

  tf.reset_default_graph()

  # Data tensors.
  features_tensor = tf.constant(x_train.astype("float32"), name="features")
  labels_tensor = tf.constant(y_train.astype("float32"), name="labels")

  # Linear model.
  weights = tf.Variable(tf.zeros(dimension, dtype=tf.float32),
                        name="weights")
  threshold = tf.Variable(0, name="threshold", dtype=tf.float32)
  predictions_tensor = (tf.tensordot(features_tensor, weights, axes=(1, 0))
                        + threshold)

  # Set up hinge loss objective.
  objective = tf.losses.hinge_loss(labels=labels_tensor,
                                   logits=predictions_tensor)

  # Set up the optimizer and get `train_op` for gradient updates.
  solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = solver.minimize(objective)

  # Start TF session and initialize variables.
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # We maintain a list of objectives and model weights during training.
  objectives = []
  models = []

  # Perform full gradient updates.
  for ii in range(loops):
    # Gradient updates.
    session.run(train_op)

    # Checkpoint once in 10 iterations.
    if ii % 10 == 0:
      # Model weights.
      model = [session.run(weights), session.run(threshold)]
      models.append(model)

      # Objective.
      objective = evaluation.expected_error_rate(
          x_train, y_train, [model], [1.0])
      objectives.append(objective)

  # Use the recorded objectives and constraints to find the best iterate.
  best_iterate = np.argmin(objectives)
  best_model = models[best_iterate]

  return best_model


def logistic_regression(train_set, learning_rate, loops):
  """Returns a model that optimizes the cross-entropy loss."""
  x_train, y_train, _ = train_set
  dimension = x_train.shape[-1]

  tf.reset_default_graph()

  # Data tensors.
  features_tensor = tf.constant(x_train.astype("float32"), name="features")
  labels_tensor = tf.constant(y_train.astype("float32"), name="labels")

  # Linear model.
  weights = tf.Variable(tf.zeros(dimension, dtype=tf.float32),
                        name="weights")
  threshold = tf.Variable(0, name="threshold", dtype=tf.float32)
  predictions_tensor = tf.tensordot(
      features_tensor, weights, axes=(1, 0)) + threshold

  # Set up cross entropy objective.
  objective = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_tensor, logits=predictions_tensor)

  # Set up the optimizer and get `train_op` for gradient updates.
  solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = solver.minimize(objective)

  # Start TF session and initialize variables.
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # We maintain a list of objectives and model weights during training.
  objectives = []
  models = []

  # Perform full gradient updates.
  for ii in range(loops):
    # Gradient updates.
    session.run(train_op)

    # Checkpoint once in 10 iterations.
    if ii % 10 == 0:
      # Model weights.
      model = [session.run(weights), session.run(threshold)]
      models.append(model)

      # Objective.
      objective = evaluation.cross_entropy_loss(
          x_train, y_train, model[0], model[1])
      objectives.append(objective)

  # Use the recorded objectives and constraints to find the best iterate.
  best_iterate = np.argmin(objectives)
  best_model = models[best_iterate]

  return best_model


def post_shift_dp(train_set, vali_set, test_set, logreg_model):
  """Post-shifts log. regression model for demographic parity using vali_set.

  Returns the train, validation and test sets with the group attribute appended
  as an additional feature, and the post-shifted linear model for the expanded
  datasets.

  Args:
    train_set: (features, labels, groups)
    vali_set: (features, labels, groups)
    test_set: (features, labels, groups)
    logreg_model: (weights, threshold)

  Returns:
    a tuple containing:
      the post_shifted model: (post_shifted_weights, post_shifted_threshold),
      the expanded training set: (features, labels, groups),
      the expanded validation set: (features, labels, groups),
      the expanded test set: (features, labels, groups).
  """
  # Compute post-shift thresholds on the *validation* set.
  x_vali, y_vali, z_vali = vali_set

  weights, threshold = logreg_model
  predictions = np.dot(x_vali, weights) + threshold
  p = y_vali.mean()  # Overall proportion of positives.

  # Set threshold for each group so that coverage for that group is equal to p.
  threshold0 = np.percentile(predictions[z_vali == 0], q=(1 - p) * 100)
  threshold1 = np.percentile(predictions[z_vali == 1], q=(1 - p) * 100)

  # Modify the logistic regression weights and threshold to include the
  # post-shift correction. Rather than having two separate thresholds for the
  # groups, we append the group attribute as a feature to the train, validation
  # and test sets, and return the expanded dataset and an expanded linear model.
  x_train, y_train, z_train = train_set
  x_test, y_test, z_test = test_set

  x_train_appended = np.concatenate([x_train, z_train.reshape(-1, 1)], axis=1)
  x_vali_appended = np.concatenate([x_vali, z_vali.reshape(-1, 1)], axis=1)
  x_test_appended = np.concatenate([x_test, z_test.reshape(-1, 1)], axis=1)

  train_set_appended = x_train_appended, y_train, z_train
  vali_set_appended = x_vali_appended, y_vali, z_vali
  test_set_appended = x_test_appended, y_test, z_test

  # Recall that thresholds are added, not subtracted from w.x during evaluation.
  post_shifted_weights = np.concatenate([weights, [-threshold1 + threshold0]])
  post_shifted_threshold = threshold - threshold0

  return ((post_shifted_weights, post_shifted_threshold), train_set_appended,
          vali_set_appended, test_set_appended)


def lagrangian_optimizer_kld(
    train_set, additive_slack, learning_rate, learning_rate_constraint, loops):
  """Implements surrogate-based Lagrangian optimizer (Algorithm 2).

  Specifically solves:
    min_{theta} sum_{G = 0, 1} KLD(p, pprG(theta))
      s.t. error_rate <= additive_slack,
    where p is the overall proportion of positives and pprG is the positive
    prediction rate for group G.

  We frame this as a constrained optimization problem:
    min_{theta, xi_pos0, xi_pos1, xi_neg0, xi_neg1} {
      -p log(xi_pos0) - (1-p) log(xi_neg0) - p log(xi_pos1)
        -(1-p) log(xi_neg1)}
    s.t.
      error_rate <= additive_slack,
        xi_pos0 <= ppr0(theta), xi_neg0 <= npr0(theta),
        xi_pos1 <= ppr1(theta), xi_neg1 <= npr1(theta),
  and formulate the Lagrangian:
    max_{lambda's >= 0} min_{xi's} {
      -p log(xi_pos0) - (1-p) log(xi_neg0) - p log(xi_pos1)
        -(1-p) log(xi_neg1)
       + lambda_pos0 (xi_pos0 - ppr0(theta))
       + lambda_neg0 (xi_neg0 - npr0(theta))
       + lambda_pos1 (xi_pos1 - ppr1(theta))
       + lambda_neg1 (xi_neg1 - npr1(theta))}
    s.t.
      error_rate <= additive_slack.

  We do best response for the slack variables xi:
    BR for xi_pos0 = p / lambda_pos0
    BR for xi_neg0 = (1 - p) / lambda_neg0
    BR for xi_pos1 = p / lambda_pos1
    BR for xi_neg1 = (1 - p) / lambda_neg1
  We do gradient ascent on the lambda's, where
    Gradient w.r.t. lambda_pos0
      = BR for xi_pos0 - ppr0(theta)
      = p / lambda_pos0 - ppr0(theta)
      = Gradient w.r.t. lambda_pos0 of
        (p log(lambda_pos0) - lambda_pos0 ppr0(theta))
    Gradient w.r.t. lambda_neg0
      = Gradient w.r.t. lambda_neg0 of
        ((1 - p) log(lambda_neg0) - lambda_neg0 npr0(theta))
    Gradient w.r.t. lambda_pos1
      = Gradient w.r.t. lambda_pos1 of
        (p log(lambda_pos1) - lambda_pos1 ppr1(theta))
    Gradient w.r.t. lambda_neg1
      = Gradient w.r.t. lambda_neg1 of
        ((1 - p) log(lambda_neg1) - lambda_neg1 npr1(theta)).
  We do gradient descent on thetas's, with ppr's and npr's replaced with hinge
  surrogates. We use concave lower bounds on ppr's and npr's, so that when they
  get negated in the updates, we get convex upper bounds.

  See Appendix D.1 in the paper for more details.

  Args:
    train_set: (features, labels, groups)
    additive_slack: float, additive slack on error rate constraint
    learning_rate: float, learning rate for model parameters
    learning_rate_constraint: float, learning rate for Lagrange multipliers
    loops: int, number of iterations

  Returns:
    stochastic_model containing list of models and probabilities,
    deterministic_model.
  """
  x_train, y_train, z_train = train_set
  dimension = x_train.shape[-1]

  tf.reset_default_graph()

  # Data tensors.
  features_tensor = tf.constant(x_train.astype("float32"), name="features")
  labels_tensor = tf.constant(y_train.astype("float32"), name="labels")

  # Linear model.
  weights = tf.Variable(tf.zeros(dimension, dtype=tf.float32),
                        name="weights")
  threshold = tf.Variable(0, name="threshold", dtype=tf.float32)
  predictions_tensor = (tf.tensordot(features_tensor, weights, axes=(1, 0))
                        + threshold)

  # Group-specific predictions.
  predictions_group0 = tf.boolean_mask(predictions_tensor, mask=(z_train < 1))
  num_examples0 = np.sum(z_train < 1)
  predictions_group1 = tf.boolean_mask(predictions_tensor, mask=(z_train > 0))
  num_examples1 = np.sum(z_train > 0)

  # We use the TF Constrained Optimization (TFCO) library to set up the
  # constrained optimization problem. The library doesn't currently support best
  # responses for slack variables. So we maintain explicit Lagrange multipliers
  # for the slack variables, and let the library deal with the Lagrange
  # multipliers for the error rate constraint.

  # Since we need to perform a gradient descent update on the model parameters,
  # and an ascent update on the Lagrange multipliers on the slack variables, we
  # create a single "minimization" objective using stop gradients, where a
  # descent gradient update has the effect of minimizing over the model
  # parameters and maximizing over the Lagrange multipliers for the slack
  # variables. As noted above, the ascent update on the Lagrange multipliers for
  # the error rate constraint is done by the library internally.

  # Placeholders for Lagrange multipliers for the four slack variables.
  lambda_pos0 = tf.Variable(0.5, dtype=tf.float32, name="lambda_pos0")
  lambda_neg0 = tf.Variable(0.5, dtype=tf.float32, name="lambda_neg0")
  lambda_pos1 = tf.Variable(0.5, dtype=tf.float32, name="lambda_pos1")
  lambda_neg1 = tf.Variable(0.5, dtype=tf.float32, name="lambda_neg1")

  # Set up prediction rates and surrogate relaxations on them.
  p = np.mean(y_train)  # Proportion of positives.

  # Positive and negative prediction rates for group 0 and group 1.
  ppr_group0 = tf.reduce_sum(tf.cast(
      tf.greater(predictions_group0, tf.zeros(num_examples0, dtype="float32")),
      "float32")) / num_examples0
  npr_group0 = 1 - ppr_group0
  ppr_group1 = tf.reduce_sum(tf.cast(
      tf.greater(predictions_group1, tf.zeros(num_examples1, dtype="float32")),
      "float32")) / num_examples1
  npr_group1 = 1 - ppr_group1

  # Hinge concave lower bounds on the positive and negative prediction rates.
  # In the gradient updates, these get negated and become convex upper bounds.
  # For group 0:
  ppr_hinge_group0 = tf.reduce_sum(
      1 - tf.nn.relu(1 - predictions_group0)) * 1.0 / num_examples0
  npr_hinge_group0 = tf.reduce_sum(
      1 - tf.nn.relu(1 + predictions_group0)) * 1.0 / num_examples0
  # For group 1:
  ppr_hinge_group1 = tf.reduce_sum(
      1 - tf.nn.relu(1 - predictions_group1)) * 1.0 / num_examples1
  npr_hinge_group1 = tf.reduce_sum(
      1 - tf.nn.relu(1 + predictions_group1)) * 1.0 / num_examples1

  # Set up KL-divergence objective for constrained optimization.
  # We use stop gradients to ensure that a single descent gradient update on the
  # objective has the effect of minimizing over the model parameters and
  # maximizing over the Lagrange multipliers for the slack variables.

  # KL-divergence for group 0.
  kld_hinge_pos_group0 = (
      - tf.stop_gradient(lambda_pos0) * ppr_hinge_group0
      - p * tf.log(lambda_pos0) + lambda_pos0 * tf.stop_gradient(ppr_group0))
  kld_hinge_neg_group0 = (
      - tf.stop_gradient(lambda_neg0) * npr_hinge_group0
      - (1 - p) * tf.log(lambda_neg0)
      + lambda_neg0 * tf.stop_gradient(npr_group0))
  kld_hinge_group0 = kld_hinge_pos_group0 + kld_hinge_neg_group0

  # KL-divergence for group 1.
  kld_hinge_pos_group1 = (
      - tf.stop_gradient(lambda_pos1) * ppr_hinge_group1
      - p * tf.log(lambda_pos1) + lambda_pos1 * tf.stop_gradient(ppr_group1))
  kld_hinge_neg_group1 = (
      - tf.stop_gradient(lambda_neg1) * npr_hinge_group1
      - (1 - p) * tf.log(lambda_neg1)
      + lambda_neg1 * tf.stop_gradient(npr_group1))
  kld_hinge_group1 = kld_hinge_pos_group1 + kld_hinge_neg_group1

  # Wrap the objective into a rate object.
  objective = tfco.wrap_rate(kld_hinge_group0 + kld_hinge_group1)

  # Set up error rate constraint for constrained optimization.
  context = tfco.rate_context(predictions_tensor, labels_tensor)
  error = tfco.error_rate(context)
  constraints = [error <= additive_slack]

  # Cretae rate minimization problem object.
  problem = tfco.RateMinimizationProblem(objective, constraints)

  # Set up optimizer.
  optimizer = tfco.LagrangianOptimizerV1(
      tf.train.AdamOptimizer(learning_rate=learning_rate),
      constraint_optimizer=tf.train.AdamOptimizer(
          learning_rate=learning_rate_constraint))
  train_op = optimizer.minimize(problem)

  # Start TF session and initialize variables.
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # We maintain a list of objectives and model weights during training.
  objectives = []
  violations = []
  models = []

  # Perform full gradient updates.
  for ii in range(loops):

    # Gradient updates.
    session.run(train_op)

    # Checkpoint once in 10 iterations.
    if ii % 10 == 0:
      # Model weights.
      model = [session.run(weights), session.run(threshold)]
      models.append(model)

      # Objective.
      klds = evaluation.expected_group_klds(
          x_train, y_train, z_train, [model], [1.0])
      objectives.append(sum(klds))

      # Violation.
      error = evaluation.expected_error_rate(
          x_train, y_train, [model], [1.0])
      violations.append([error - additive_slack])

  # Use the recorded objectives and constraints to find the best iterate.
  best_iterate = tfco.find_best_candidate_index(
      np.array(objectives), np.array(violations))
  deterministic_model = models[best_iterate]

  # Use shrinking to find a sparse distribution over iterates.
  probabilities = tfco.find_best_candidate_distribution(
      np.array(objectives), np.array(violations))
  models_pruned = [models[i] for i in range(len(models)) if
                   probabilities[i] > 0.0]
  probabilities_pruned = probabilities[probabilities > 0.0]

  return (models_pruned, probabilities_pruned), deterministic_model
