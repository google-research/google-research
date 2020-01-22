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


class NonScalingDropout(tf.keras.layers.Dropout):
  """Applies dropout to the `inputs` without rescaling the kept elements.

  Dropout consists in randomly setting a fraction of input units to 0 at each
  update during training time, which helps prevent overfitting. The units that
  are kept are not scaled.
  """

  def __init__(self,
               rate,
               noise_shape=None,
               seed=None,
               training=False,
               **kwargs):
    """Initializes the layer.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D tensor of type `int32` representing the shape of the
        binary dropout mask that will be multiplied with the input. For
        instance, if your inputs have shape `[batch_size, timesteps, features]`,
        and you want the dropout mask to be the same for all timesteps, you can
        use `noise_shape=[batch_size, 1, features]`.
      seed: Used to create random seeds. See `tf.set_random_seed` for behavior.
        or in inference mode (return the input untouched).
      training: Boolean, indicating whether the layer is created for training
        or inference.
      **kwargs: Keword arguments
    """
    super(NonScalingDropout, self).__init__(rate, noise_shape, seed, **kwargs)
    self.training = training

  def call(self, inputs):
    if not self.training or self.rate == 0:
      return inputs
    else:
      if self.noise_shape is None:
        self.noise_shape = tf.shape(inputs)
      noise_mask = tf.keras.backend.random_uniform(
          self.noise_shape, seed=self.seed) < (1 - self.rate)
      return inputs * tf.keras.backend.cast(noise_mask, tf.float32)
