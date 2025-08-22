# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Feature selection with Orthogonal Matching Pursuit."""

from sequential_attention.benchmarks.orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from sequential_attention.experiments.models.mlp import MLPModel
import tensorflow as tf


class OrthogonalMatchingPursuitModel(MLPModel):
  """MLP with Orthogonal Matching Pursuit."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_train_steps,
      num_inputs_to_select_per_step=1,
      **kwargs,
  ):
    """Initialize the model."""

    super(OrthogonalMatchingPursuitModel, self).__init__(**kwargs)

    self.omp = OrthogonalMatchingPursuit(
        num_inputs,
        num_inputs_to_select,
        num_inputs_to_select_per_step=num_inputs_to_select_per_step,
    )

    self.num_inputs = num_inputs
    self.num_train_steps = num_train_steps

  def call(self, inputs, training=False, omp_attention=True):
    training_percentage = self.optimizer.iterations / self.num_train_steps
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    if omp_attention:
      feature_weights = self.omp(training_percentage)
    else:
      feature_weights = tf.ones(self.num_inputs)
    inputs = tf.multiply(inputs, feature_weights)
    representation = self.mlp_model(inputs)
    prediction = self.mlp_predictor(representation)
    return prediction

  def train_step(self, inputs):
    """Custom train step using the `compute_loss` method."""

    x, y = inputs
    with tf.GradientTape() as tape:
      y_pred = self.call(x, training=True)
      loss = self.compute_loss(x, y, y_pred)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Update OMP gradients
    with tf.GradientTape() as tape:
      y_pred_omp = self.call(x, training=True, omp_attention=False)
      omp_loss = self.compute_loss(x, y, y_pred_omp)
    gradients = tape.gradient(omp_loss, self.mlp_model.weights[0])
    gradients = tf.norm(gradients, axis=1)
    assign_gradient = self.omp.gradients.assign(gradients)

    with tf.control_dependencies([assign_gradient]):  # force update
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}
