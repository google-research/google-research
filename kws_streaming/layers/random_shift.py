# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Augment audio data with random shifts."""

from kws_streaming.layers.compat import tf
from tensorflow.python.keras.utils import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import


@tf.function
def random_shift(inputs, time_shift, seed=None):
  """Shifts input data randomly in time dim.

  It can be useful for augmenting training data with random shifts in time dim
  for making model more robust to input audio shifts

  Args:
    inputs: input tensor [batch_size, time]
    time_shift: defines time shift range: -time_shift...time_shift
      it is defiend in samples
    seed: random seed
  Returns:
    masked image
  Raises:
    ValueError: if inputs.shape.rank != 2
  """
  if inputs.shape.rank != 2:
    raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

  inputs_shape = inputs.shape.as_list()
  batch_size = inputs_shape[0]
  sequence_length = inputs_shape[1]

  # below function will process 2D arrays, convert it to [batch, time, dummy]
  inputs = tf.expand_dims(inputs, 2)

  time_shift_amounts = tf.random.uniform(
      shape=[batch_size],
      minval=-time_shift,
      maxval=time_shift,
      dtype=tf.int32,
      seed=seed)

  outputs = tf.TensorArray(inputs.dtype, 0, dynamic_size=True)
  for i in tf.range(batch_size):
    time_shift_amount = time_shift_amounts[i]

    # pylint: disable=cell-var-from-loop
    time_shift_padding = tf.cond(time_shift_amount > 0,
                                 lambda: [[time_shift_amount, 0], [0, 0]],
                                 lambda: [[0, -time_shift_amount], [0, 0]])
    time_shift_offset = tf.cond(time_shift_amount > 0, lambda: [0, 0],
                                lambda: [-time_shift_amount, 0])
    # pylint: enable=cell-var-from-loop

    padded = tf.pad(
        tensor=inputs[i], paddings=time_shift_padding, mode='CONSTANT')
    padded_sliced = tf.slice(padded, time_shift_offset, [sequence_length, -1])

    outputs = outputs.write(i, padded_sliced)

  # convert it back to [batch, time]
  outputs = tf.squeeze(outputs.stack(), axis=[2])
  outputs.set_shape(inputs_shape)
  return outputs


class RandomShift(tf.keras.layers.Layer):
  """Randomly shifts data in time dim.

  It can be useful for augmenting training data with random shifts in time dim
  to makw model more robust to input audio shifts also can be useful
  for generating more training data.

  Attributes:
    time_shift: defines time shift range: -time_shift...time_shift
      it is defiend in samples
    seed: random seed
    **kwargs: additional layer arguments
  """

  def __init__(self,
               time_shift=0,
               seed=None,
               **kwargs):
    super().__init__(**kwargs)
    self.time_shift = time_shift
    self.seed = seed

  def call(self, inputs, training=None):

    if inputs.shape.rank != 2:  # [batch, time]
      raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

    if not self.time_shift:
      return inputs

    if training is None:
      training = tf.keras.backend.learning_phase()
    # pylint: disable=g-long-lambda
    return control_flow_util.smart_cond(
        training, lambda: random_shift(
            inputs,
            self.time_shift,
            self.seed), lambda: array_ops.identity(inputs))
    # pylint: enable=g-long-lambda

  def get_config(self):
    config = {
        'time_shift': self.time_shift,
        'seed': self.seed,
    }
    base_config = super(RandomShift, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
