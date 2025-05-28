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

# This file is branched from 'tensorflow_addons/layers/crf.py'
# As of 2022-11-15, the usage of `tf.cond` in this file does not support JIT
# compile when using TPU. So they are removed, as for our case, the sequencce
# length is always > 1.
"""Implementing Conditional Random Field layer."""
import tensorflow as tf

from tensorflow_addons.utils import types

from mave.benchmark.models import crf_utils


def _compute_mask_right_boundary(mask):
  """input mask: 0011100, output right_boundary: 0000100."""
  # shift mask to left by 1: 0011100 => 0111000
  offset = 1
  left_shifted_mask = tf.concat(
      [mask[:, offset:], tf.zeros_like(mask[:, :offset])], axis=1)

  # NOTE: below code is different from keras_contrib
  # Original code in keras_contrib:
  # end_mask = K.cast(
  #   K.greater(self.shift_left(mask), mask),
  #   K.floatx()
  # )
  # has a bug, confirmed
  # by the original keras_contrib maintainer
  # Luiz Felix (github: lzfelix),

  # 0011100 > 0111000 => 0000100
  right_boundary = tf.math.greater(
      tf.cast(mask, tf.int32), tf.cast(left_shifted_mask, tf.int32))

  return right_boundary


def _compute_mask_left_boundary(mask):
  """input mask: 0011100, output left_boundary: 0010000."""
  # shift mask to right by 1: 0011100 => 0001110
  offset = 1
  right_shifted_mask = tf.concat(
      [tf.zeros_like(mask[:, :offset]), mask[:, :-offset]], axis=1)

  # 0011100 > 0001110 => 0010000
  left_boundary = tf.math.greater(
      tf.cast(mask, tf.int32), tf.cast(right_shifted_mask, tf.int32))

  return left_boundary


class CRF(tf.keras.layers.Layer):
  """Linear chain conditional random field (CRF).

    Inherits from: `tf.keras.layers.Layer`.

    References:
        - [Conditional Random
        Field](https://en.wikipedia.org/wiki/Conditional_random_field)

    Example:

    >>> layer = tfa.layers.CRF(4)
    >>> inputs = np.random.rand(2, 4, 8).astype(np.float32)
    >>> decoded_sequence, potentials, sequence_length, chain_kernel =
    layer(inputs)
    >>> decoded_sequence.shape
    TensorShape([2, 4])
    >>> potentials.shape
    TensorShape([2, 4, 4])
    >>> sequence_length
    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([4, 4])>
    >>> chain_kernel.shape
    TensorShape([4, 4])

    Args:
        units: Positive integer, dimensionality of the reservoir.
        chain_initializer: Orthogonal matrix. Default to `orthogonal`.
        use_boundary: `Boolean`, whether the layer uses a boundary vector.
          Default to `True`.
        boundary_initializer: Tensors initialized to 0. Default to `zeros`.
        use_kernel: `Boolean`, whether the layer uses a kernel weights. Default
          to `True`.

    Args:
        inputs: Positive integer, dimensionality of the output space.
        mask: A boolean `Tensor` of shape `[batch_size, sequence_length]` or
          `None`. Default to `None`.

    Raises:
        ValueError: If input mask doesn't have dim 2 or None.
        NotImplementedError: If left padding is provided.
  """

  def __init__(
      self,
      units,
      chain_initializer = "orthogonal",
      use_boundary = True,
      boundary_initializer = "zeros",
      use_kernel = True,
      **kwargs,
  ):
    super().__init__(**kwargs)

    # setup mask supporting flag, used by base class (the Layer)
    # because base class's init method will set it to False unconditionally
    # So this assigned must be executed after call base class's init method
    self.supports_masking = True

    self.units = units  # numbers of tags

    self.use_boundary = use_boundary
    self.use_kernel = use_kernel
    self.chain_initializer = tf.keras.initializers.get(chain_initializer)
    self.boundary_initializer = tf.keras.initializers.get(boundary_initializer)

    # weights that work as transfer probability of each tags
    self.chain_kernel = self.add_weight(
        shape=(self.units, self.units),
        name="chain_kernel",
        initializer=self.chain_initializer,
    )

    # weight of <START> to tag probability and tag to <END> probability
    if self.use_boundary:
      self.left_boundary = self.add_weight(
          shape=(self.units,),
          name="left_boundary",
          initializer=self.boundary_initializer,
      )
      self.right_boundary = self.add_weight(
          shape=(self.units,),
          name="right_boundary",
          initializer=self.boundary_initializer,
      )

    if self.use_kernel:
      self._dense_layer = tf.keras.layers.Dense(
          units=self.units, dtype=self.dtype)
    else:
      self._dense_layer = lambda x: tf.cast(x, dtype=self.dtype)

  def call(self, inputs, mask=None):
    # mask: Tensor(shape=(batch_size, sequence_length), dtype=bool) or None

    if mask is not None:
      if tf.keras.backend.ndim(mask) != 2:
        raise ValueError("Input mask to CRF must have dim 2 if not None")

    if mask is not None:
      # left padding of mask is not supported, due the underline CRF function
      # detect it and report it to user
      left_boundary_mask = _compute_mask_left_boundary(mask)
      first_mask = left_boundary_mask[:, 0]
      if first_mask is not None and tf.executing_eagerly():
        no_left_padding = tf.math.reduce_all(first_mask)
        left_padding = not no_left_padding
        if left_padding:
          raise NotImplementedError(
              "Currently, CRF layer do not support left padding")

    potentials = self._dense_layer(inputs)

    # appending boundary probability info
    if self.use_boundary:
      potentials = self.add_boundary_energy(potentials, mask,
                                            self.left_boundary,
                                            self.right_boundary)

    sequence_length = self._get_sequence_length(inputs, mask)

    decoded_sequence, _ = self.get_viterbi_decoding(potentials, sequence_length)

    return [decoded_sequence, potentials, sequence_length, self.chain_kernel]

  def _get_sequence_length(self, input_, mask):
    """Returns the sequence length from input and mask."""
    if mask is not None:
      sequence_length = self.mask_to_sequence_length(mask)
    else:
      # make a mask tensor from input, then used to generate sequence_length
      input_energy_shape = tf.shape(input_)
      raw_input_shape = tf.slice(input_energy_shape, [0], [2])
      alt_mask = tf.ones(raw_input_shape)

      sequence_length = self.mask_to_sequence_length(alt_mask)

    return sequence_length

  def mask_to_sequence_length(self, mask):
    """compute sequence length from mask."""
    sequence_length = tf.reduce_sum(tf.cast(mask, tf.int64), 1)
    return sequence_length

  def add_boundary_energy(self, potentials, mask, start, end):

    def expand_scalar_to_3d(x):
      # expand tensor from shape (x, ) to (1, 1, x)
      return tf.reshape(x, (1, 1, -1))

    start = tf.cast(expand_scalar_to_3d(start), potentials.dtype)
    end = tf.cast(expand_scalar_to_3d(end), potentials.dtype)
    if mask is None:
      potentials = tf.concat(
          [potentials[:, :1, :] + start, potentials[:, 1:, :]], axis=1)
      potentials = tf.concat(
          [potentials[:, :-1, :], potentials[:, -1:, :] + end], axis=1)
    else:
      mask = tf.keras.backend.expand_dims(tf.cast(mask, start.dtype), axis=-1)
      start_mask = tf.cast(_compute_mask_left_boundary(mask), start.dtype)

      end_mask = tf.cast(_compute_mask_right_boundary(mask), end.dtype)
      potentials = potentials + start_mask * start
      potentials = potentials + end_mask * end
    return potentials

  def get_viterbi_decoding(self, potentials, sequence_length):
    # decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`
    decode_tags, best_score = crf_utils.crf_decode(potentials,
                                                   self.chain_kernel,
                                                   sequence_length)

    return decode_tags, best_score

  def get_config(self):
    # used for loading model from disk
    config = {
        "units":
            self.units,
        "chain_initializer":
            tf.keras.initializers.serialize(self.chain_initializer),
        "use_boundary":
            self.use_boundary,
        "boundary_initializer":
            tf.keras.initializers.serialize(self.boundary_initializer),
        "use_kernel":
            self.use_kernel,
    }
    base_config = super().get_config()
    return {**base_config, **config}

  def compute_output_shape(self, input_shape):
    output_shape = input_shape[:2]
    return output_shape

  def compute_mask(self, input_, mask=None):
    """keep mask shape [batch_size, max_seq_len]."""
    return mask

  @property
  def _compute_dtype(self):
    # fixed output dtype from underline CRF functions
    return tf.int32
