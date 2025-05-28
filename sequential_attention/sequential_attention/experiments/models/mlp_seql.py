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

"""Feature selection with Sequential LASSO."""

from sequential_attention.benchmarks.sequential_lasso import SequentialLassoRegularizer
from sequential_attention.experiments.models.mlp import MLPModel
import tensorflow as tf


class SequentialLASSOModel(MLPModel):
  """MLP with Sequential Lasso."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_train_steps,
      num_inputs_to_select_per_step=1,
      layer_sequence=None,
      alpha=0,
      group_lasso_scale=0.01,
      **kwargs,
  ):
    """Initialize the model."""

    super(SequentialLASSOModel, self).__init__(**kwargs)

    # first layer
    init_kernel = tf.random.normal(
        shape=[num_inputs, layer_sequence[0]], stddev=0.001, dtype=tf.float32
    )
    self.kernel0 = tf.Variable(
        initial_value=lambda: init_kernel, dtype=tf.float32, name='kernel'
    )
    self.bias0 = tf.Variable(
        initial_value=tf.zeros([layer_sequence[0]]), dtype=tf.float32
    )
    # other layers
    mlp_sequence = [
        tf.keras.layers.Dense(
            dim, activation=tf.keras.layers.LeakyReLU(alpha=alpha)
        )
        for dim in layer_sequence[1:]
    ]
    self.mlp_model = tf.keras.Sequential(mlp_sequence)

    self.seql = SequentialLassoRegularizer(
        num_inputs=num_inputs,
        num_inputs_to_select=num_inputs_to_select,
        num_inputs_to_select_per_step=num_inputs_to_select_per_step,
        group_lasso_scale=group_lasso_scale,
    )
    self.num_train_steps = num_train_steps

  def call(self, inputs, training=False):
    if self.batch_norm:
      inputs = self.batch_norm_layer(inputs, training=training)
    inputs = tf.linalg.matmul(inputs, self.kernel0) + self.bias0  # first layer
    representation = self.mlp_model(inputs)  # other layers
    prediction = self.mlp_predictor(representation)
    return prediction

  def train_step(self, inputs):
    """Custom train step."""
    training_percentage = self.optimizer.iterations / self.num_train_steps

    with tf.GradientTape() as tape:
      x, y = inputs
      y_pred = self.call(x, training=True)
      norms = tf.norm(self.kernel0, axis=1)
      reg = self.seql(norms, training_percentage=training_percentage)
      loss = self.compute_loss(x, y, y_pred) + reg
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
