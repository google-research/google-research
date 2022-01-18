# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
  max F-measure s.t. F-measure(group1) >= F-measure(group0) - epsilon.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from generalized_rates.fmeasure_optimization import evaluation
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


def post_shift_fmeasure(vali_set, logreg_model, bin_size=0.001):
  """Post-shifts logistic regression model to optimize F-measure."""
  x_vali, y_vali, _ = vali_set

  predictions = np.dot(x_vali, logreg_model[0]) + logreg_model[1]
  threshold_candidates = np.percentile(
      np.unique(predictions), q=np.arange(0, 100, 100 * bin_size))

  best_fm = -1
  best_threshold = -1
  for threshold in threshold_candidates:
    fm = evaluation.fmeasure_for_predictions(y_vali, predictions - threshold)
    if fm > best_fm:
      best_fm = fm
      best_threshold = threshold

  best_weights = logreg_model[0]
  best_threshold = logreg_model[1] - best_threshold

  return (best_weights, best_threshold)


def lagrangian_optimizer_fmeasure(
    train_set, epsilon, learning_rate, learning_rate_constraint, loops):
  """Implements surrogate-based Lagrangian optimizer (Algorithm 3).

  Specifically solves:
    max F-measure s.t. F-measure(group1) >= F-measure(group0) - epsilon.

  Args:
    train_set: (features, labels, groups)
    epsilon: float, constraint slack.
    learning_rate: float, learning rate for model parameters.
    learning_rate_constraint: float, learning rate for Lagrange multipliers.
    loops: int, number of iterations.

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

  # Contexts.
  context = tfco.rate_context(predictions_tensor, labels_tensor)
  context0 = context.subset(z_train < 1)
  context1 = context.subset(z_train > 0)

  # F-measure rates.
  fm_overall = tfco.f_score(context)
  fm1 = tfco.f_score(context1)
  fm0 = tfco.f_score(context0)

  # Rate minimization problem.
  problem = tfco.RateMinimizationProblem(-fm_overall, [fm0 <= fm1 + epsilon])

  # Optimizer.
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
      objective = -evaluation.expected_fmeasure(
          x_train, y_train, [model], [1.0])
      objectives.append(objective)

      # Violation.
      fmeasure0, fmeasure1 = evaluation.expected_group_fmeasures(
          x_train, y_train, z_train, [model], [1.0])
      violations.append([fmeasure0 - fmeasure1 - epsilon])

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
