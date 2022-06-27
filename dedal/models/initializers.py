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

"""Implements custom initializers."""

from typing import Any, Dict, Mapping, Optional, Union

import gin
import numpy as np
import tensorflow as tf

from dedal import vocabulary

BLOSUM_62 = """
A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""
gin.constant('BLOSUM_62', BLOSUM_62)


Initializer = Union[tf.initializers.Initializer, str]


@gin.configurable
class HarmonicEmbeddings(tf.initializers.Initializer):
  """Initializes weights for sinusoidal positional embeddings.

  Attributes:
    scale_factor: angular frequencies for sinusoidal embeddings will be
      logarithmically spaced between max_freq x scale_factor and max_freq,
      with base equal to scale_factor.
    max_freq: the largest angular frequency to be used.
  """

  def __init__(
      self, scale_factor = 1e-4, max_freq = 1.0, **kwargs):
    super().__init__(**kwargs)
    self._scale_factor = scale_factor
    self._max_freq = max_freq

  def __call__(
      self,
      shape,
      dtype = tf.float32,
  ):
    if len(shape) != 2:
      raise ValueError('shape must have length two.')
    max_len, emb_dim = shape
    if emb_dim % 2:
      raise ValueError('dimension of embeddings must be even.')
    n_freqs = emb_dim // 2

    pos = tf.range(max_len, dtype=dtype)
    ang_freq = self._max_freq * tf.experimental.numpy.logspace(
        0.0, 1.0, n_freqs, base=self._scale_factor, dtype=dtype)
    phase = pos[:, None] * ang_freq[None, :]
    embeddings = tf.concat(
        (tf.sin(phase)[:, :, None], tf.cos(phase)[:, :, None]), -1)
    return tf.reshape(embeddings, (max_len, -1))

  def get_config(self):
    config = super().get_config()
    config.update({
        'scale_factor': self._scale_factor,
        'max_freq': self._max_freq,
    })
    return config


@gin.configurable
class SubsMatInitializer(tf.initializers.Initializer):
  """Initializes amino acid substitution matrix."""

  def __init__(
      self,
      vocab = None,
      matrix_str = BLOSUM_62,
      pad_penalty = -1e9,
      **kwargs):
    super().__init__(**kwargs)
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._matrix_str = matrix_str
    self._pad_penalty = pad_penalty

  def _load_subs_mat(self):
    """Loads FASTA36's substitution matrix file into Python dict format."""
    subs_mat = {}
    tokens = None
    for line in self._matrix_str.split('\n'):
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      if tokens is None:  # Processes header.
        tokens = line.split()
        continue

      token, vals = line.split(maxsplit=1)  # Processes row.
      subs_mat[token] = {k: float(v) for k, v in zip(tokens, vals.split())}
    return subs_mat

  def _fill_gaps(self, weights):
    """Sets the gap penalties in the input weight matrix of size |V|x|V|."""
    for token in self._vocab.get_specials() + (self._vocab.MASK,):
      idx = self._vocab.get(token)
      weights[idx] = self._pad_penalty
      weights[:, idx] = self._pad_penalty

  def _from_file(self, dtype = tf.float32):
    """Creates subs matrix tf.Tensor from FASTA36's substitution matrix file."""
    subs_mat = self._load_subs_mat()
    voc_size = len(self._vocab)
    result = np.empty((voc_size, voc_size), dtype=dtype.as_numpy_dtype())
    self._fill_gaps(result)
    for token_i in self._vocab.tokens:
      idx_i = self._vocab.get(token_i)
      token_i = token_i if token_i in subs_mat else 'X'
      for token_j in self._vocab.tokens:
        idx_j = self._vocab.get(token_j)
        token_j = token_j if token_j in subs_mat else 'X'
        result[idx_i][idx_j] = subs_mat[token_i][token_j]

    return tf.convert_to_tensor(result, dtype=dtype)

  def _random_sample(self, dtype = tf.float32):
    # TODO(fllinares): move away from NumPy's PRNG to TensorFlow's
    # TODO(fllinares): take into account subs matrix "biological" constraints
    voc_size = len(self._vocab)
    result = np.random.randn(voc_size, voc_size)
    result = 0.5 * (result + result.T)
    self._fill_gaps(result)
    return tf.convert_to_tensor(result, dtype=dtype)

  def __call__(self,
               shape,
               dtype = tf.float32):
    n_tokens = len(self._vocab)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] != n_tokens:
      raise ValueError(f'Shape {shape}. incompatibility with vocabulary.')

    return (self._random_sample(dtype) if self._matrix_str is None
            else self._from_file(dtype))


@gin.configurable
class SymmetricKernelInitializer(tf.initializers.Initializer):
  """Initializes 2D symmetric kernel for bilinear form layers.

  Attributes:
    base_init: a Keras initializer to sample the possibly asymmetric kernel.
    factorized: whether to transform the kernel returned by base_init via
      "factorization", W <- W W^{T}, or not, W <- 0.5 (W + W^{T}).
  """

  def __init__(
      self,
      base_init = 'GlorotUniform',
      factorized = True,
      **kwargs):
    super().__init__(**kwargs)
    self.base_init = tf.keras.initializers.get(base_init)
    self.factorized = factorized

  def __call__(
      self,
      shape,
      dtype = tf.float32,
  ):
    if len(shape) != 2:
      raise ValueError('shape must have length two.')
    kernel = self.base_init(shape, dtype)
    return (tf.matmul(kernel, kernel, transpose_b=True) if self.factorized
            else 0.5 * (kernel + tf.transpose(kernel)))

  def get_config(self):
    config = super().get_config()
    config.update({
        'base_init': tf.keras.initializers.serialize(self.base_init),
        'factorized': self.factorized,
    })
    return config
