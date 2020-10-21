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

"""Dropout layer which doesn't rescale the kept elements."""

from kws_streaming.layers.compat import tf
from tensorflow.python.keras.utils import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import
# TODO(b/171351822) migrate to tf v2 and improve imported dependency


class NonScalingDropout(tf.keras.layers.Dropout):
  """Applies dropout to the `inputs` without rescaling the kept elements.

  Dropout consists in randomly setting a fraction of input units to 0 at each
  update during training time, which helps prevent overfitting. The units that
  are kept are not scaled.
  """

  def call(self, inputs, training=None):

    if self.rate == 0.0:
      return inputs

    if training is None:
      training = tf.keras.backend.learning_phase()

    if self.noise_shape is None:
      self.noise_shape = tf.shape(inputs)

    return control_flow_util.smart_cond(
        training, lambda: self._non_scaling_drop_op(inputs),
        lambda: array_ops.identity(inputs))

  def _non_scaling_drop_op(self, inputs):
    return inputs * tf.keras.backend.cast(
        tf.keras.backend.random_uniform(self.noise_shape, seed=self.seed) <
        (1 - self.rate), tf.float32)
