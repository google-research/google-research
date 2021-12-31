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

"""Transformations to be applied on sequences."""

import abc
from typing import Dict, Optional, Sequence, Tuple, Union

import gin
import tensorflow as tf

from dedal import pairs
from dedal import vocabulary

Keys = Union[str, Sequence[str]]
Example = Dict[str, tf.Tensor]


@gin.configurable
class Transform(abc.ABC):
  """A generic class for transformations."""

  def __init__(self,
               on = 'sequence',
               out = None,
               vocab = None):
    self._on = (on,) if isinstance(on, str) else on
    out = self._on if out is None else out
    self._out = (out,) if isinstance(out, str) else out
    self._vocab = vocabulary.get_default() if vocab is None else vocab

  def single_call(self, arg):
    raise NotImplementedError()

  def call(self, *args):
    """Assumes the same order as `on` and `out` for args and outputs.

    This method by default calls the `single_call` method over each argument.
    For Transforms over single argument, the `single_call` method should be
    overwritten. For Transforms over many arguments, one should directly
    overload the `call` method itself.

    Args:
      *args: the argument of the transformation. For Transform over a single
        input, it can be a Sequence of arguments, in which case the Transform
        will be applied over each of them.

    Returns:
      A tf.Tensor or tuple of tf.Tensor.
    """
    result = tuple(self.single_call(arg) for arg in args)
    return result if len(args) > 1 else result[0]

  def __call__(self, inputs):
    keys = set(inputs.keys())
    if not keys.issuperset(self._on):
      raise ValueError(f'The keys of the input ({keys}) are not matching the'
                       f' transform input keys: {self._on}')

    args = tuple(inputs.pop(key) for key in self._on)
    outputs = self.call(*args)
    outputs = (outputs,) if not isinstance(outputs, Sequence) else outputs
    for key, output in zip(self._out, outputs):
      if output is not None:
        inputs[key] = output
    for i, key in enumerate(self._on):
      if key not in self._out:
        inputs[key] = args[i]
    return inputs


@gin.configurable
class Pop(Transform):

  def call(self, *args):
    return None


@gin.configurable
class Reshape(Transform):
  """Reshapes a tensor to a compatible target shape."""

  def __init__(self, shape, **kwargs):
    super().__init__(**kwargs)
    self._shape = shape

  def single_call(self, tensor):
    return tf.reshape(tensor, self._shape)


@gin.configurable
class Stack(Transform):
  """Stacks tensors of compatible shapes."""

  def __init__(self, axis = 0, **kwargs):
    super().__init__(**kwargs)
    self._axis = axis

  def call(self, *args):
    return tf.stack(args, axis=self._axis)


@gin.configurable
class Encode(Transform):
  """Encodes a string into a integers."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(self._vocab._indices.keys())),
        values=tf.constant(list(self._vocab._indices.values()), dtype=tf.int64))
    self._lookup = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)

  def single_call(self, sequence):
    return tf.cast(self._lookup[tf.strings.bytes_split(sequence)], tf.int32)


@gin.configurable
class Recode(Transform):
  """Changes the integer-based encoding of amino acids."""

  def __init__(self, target, **kwargs):
    super().__init__(**kwargs)
    self._vocab_map = self._vocab.translate(target)

  def single_call(self, sequence):
    return tf.gather(self._vocab_map, sequence)


@gin.configurable
class CropOrPad(Transform):
  """Crops or left/right pads a sequence with the same token."""

  def __init__(self,
               size = gin.REQUIRED,
               random = True,
               right = True,
               token = None,
               seed = None,
               **kwargs):
    super().__init__(**kwargs)
    self._size = size
    self._random = random
    self._right = right
    self._token = self._vocab.get(token, self._vocab.padding_code)
    self._seed = seed

  def single_call(self, sequence):
    seq_len = tf.shape(sequence)[0]
    if seq_len < self._size:
      to_pad = self._size - seq_len
      pattern = [0, to_pad] if self._right else [to_pad, 0]
      sequence = tf.pad(sequence, [pattern], constant_values=self._token)
    elif seq_len > self._size:
      sequence = (
          tf.image.random_crop(sequence, [self._size], seed=self._seed)
          if self._random else sequence[:self._size])
    sequence.set_shape([self._size])
    return sequence


@gin.configurable
class AppendToken(Transform):
  """Left/Right pads a sequence with the a single token."""

  def __init__(self, right = True, token = None, **kwargs):
    super().__init__(**kwargs)
    self._token = self._vocab.get(token, self._vocab.padding_code)
    self._right = right

  def single_call(self, sequence):
    pattern = [0, 1] if self._right else [1, 0]
    return tf.pad(sequence, [pattern], constant_values=self._token)


@gin.configurable
class EOS(AppendToken):
  """Adds EOS token."""

  def __init__(self, token=None, **kwargs):
    """If token is not passed, assumed to be the last special."""
    super().__init__(right=True, token=token, **kwargs)  # Sets vocab and token.
    if token is None:  # Resets self._token to be the last special.
      token = self._vocab.specials[-1]
      self._token = self._vocab.get(token)


@gin.configurable
class PrependClass(AppendToken):
  """Prepends CLS token."""

  def __init__(self, token=None, **kwargs):
    """If token is not passed, assumed to be the penultimate special."""
    super().__init__(right=False, token=token, **kwargs)
    if token is None:  # Resets self._token to be the penultimate special.
      token = self._vocab.specials[-2]
      self._token = self._vocab.get(token)


@gin.configurable
class CropOrPadND(Transform):
  """Left/Right pads a tensor along its last dimension with the same value."""

  def __init__(self,
               size = 1,
               right = True,
               value = 0,
               axis = -1,
               **kwargs):
    super().__init__(**kwargs)
    self._size = size
    self._right = right
    self._axis = axis
    self._value = value

  def single_call(self, tensor):
    length = tf.shape(tensor)[self._axis]
    shape = tensor.shape.as_list()
    shape[self._axis] = self._size
    if length < self._size:
      to_pad = self._size - length
      pattern = int(tensor.shape.rank) * [[0, 0]]
      pattern[self._axis] = [0, to_pad] if self._right else [to_pad, 0]
      result = tf.pad(tensor, pattern, constant_values=self._value)
    else:
      result = tf.image.random_crop(tensor, shape)
    result.set_shape(shape)
    return result


@gin.configurable
class RemoveTokens(Transform):
  """Removes all positions in a sequence containing certain tokens."""

  def __init__(self, tokens = '-', **kwargs):
    super().__init__(**kwargs)
    self._tokens = (tokens,) if isinstance(tokens, str) else tokens

  def single_call(self, sequence):
    mask = self._vocab.compute_mask(sequence, self._tokens)
    keep_indices = tf.reshape(tf.where(mask), [-1])
    return tf.gather(sequence, keep_indices)


@gin.configurable
class OneHot(Transform):
  """Turns sequence of integers into sequences of one-hot encodings."""

  def __init__(self, depth = None, **kwargs):
    super().__init__(**kwargs)
    self._depth = depth if depth is not None else len(self._vocab)

  def single_call(self, sequence):
    return tf.one_hot(sequence, self._depth, dtype=tf.float32)


@gin.configurable
class ContactMatrix(Transform):
  """Process the 3D positions and turn them into a contact matrix."""

  def __init__(self, threshold = 10.0, **kwargs):
    super().__init__(**kwargs)
    self._threshold = threshold

  def single_call(self, positions):
    """Expects positions to be a tf.Tensor<float>[n, 3]."""
    # Makes a batch of size 1 to be compatible with pairwise_square_dist.
    pos = tf.expand_dims(positions, 0)
    sq_dist = pairs.square_distances(pos, pos)[0]
    return tf.cast(sq_dist < self._threshold ** 2, dtype=tf.float32)


@gin.configurable
class BackboneAngleTransform(Transform):
  """Maps tf.Tensor<float>[len, 1] of angles to [len, 2] tensor of sin/cos."""

  def single_call(self, angles):
    sin, cos = tf.sin(angles), tf.cos(angles)
    return tf.concat([sin, cos], -1)


@gin.configurable
class DatasetTransform(abc.ABC):
  """A generic class for dataset-level transformations."""

  @abc.abstractmethod
  def call(self, ds):
    """A transformation function to be used with `tf.data.Dataset.apply`."""
    pass

  def __call__(self, ds):
    return self.call(ds)


@gin.configurable
class FilterByLength(DatasetTransform):
  """Filters a dataset by the length of one or more sequences."""

  def __init__(self,
               on = 'seq_len',
               vocab = None,
               max_len = 512,
               precomputed = True):
    self._on = (on,) if isinstance(on, str) else on
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._max_len = max_len
    self._precomputed = precomputed

  def _get_len(self, t):
    if not self._precomputed:
      if t.dtype == tf.string:
        t = tf.reshape(t, ())
        return tf.strings.length(t)
      else:
        mask = tf.logical_and(
            self._vocab.padding_mask(t), self._vocab.special_token_mask(t))
        return tf.reduce_sum(tf.cast(mask, tf.int32))
    return t

  def _filter_fn(self, ex):
    cond = tf.convert_to_tensor(True, tf.bool)
    for key in self._on:
      cond = tf.math.logical_and(cond, self._get_len(ex[key]) <= self._max_len)
    return cond

  def call(self, ds):
    return ds.filter(self._filter_fn)
