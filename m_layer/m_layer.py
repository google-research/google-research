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

# coding=utf-8
"""Code for creating the M-layer as a keras layer."""

import tensorflow as tf


class MLayer(tf.keras.layers.Layer):
  """The M-layer: Lie Algebra generator-embedding and matrix exponentiation.

  This is a Keras implementation of the M-layer described in (2020)[1].

  #### References

  [1]: [Thomas Fischbacher, Iulia M. Comsa, Krzysztof Potempa, Moritz Firsching,
  Luca Versari, Jyrki Alakuijala "Intelligent Matrix Exponentiation",
  arxiv:2008.03936.](https://arxiv.org/abs/2008.03936)
  """

  def __init__(self,
               dim_m,
               matrix_init=None,
               with_bias=False,
               matrix_squarings_exp=None,
               **kwargs):
    """Initializes the instance.

    Args:
      dim_m: The matrix to be exponentiated in the M-layer has the shape (dim_m,
        dim_m).
      matrix_init: What initializer to use for the matrix. `None` defaults to
        `normal` initialization.
      with_bias: Whether a bias should be included in layer after
        exponentiation.
      matrix_squarings_exp: None to compute tf.linalg.expm(M), an integer `k` to
        instead approximate it with (I+M/2**k)**(2**k).
      **kwargs: keyword arguments passed to the Keras layer base class.
    """
    self._dim_m = dim_m
    self._rep_to_exp_tensor = None
    self._matrix_init = matrix_init or 'uniform'
    self._with_bias = with_bias
    self._matrix_bias = None
    self._matrix_squarings_exp = matrix_squarings_exp
    super(MLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    dim_rep = input_shape[-1]
    self._rep_to_exp_tensor = self.add_weight(
        name='rep_to_exp_tensor',
        shape=(dim_rep, self._dim_m, self._dim_m),
        initializer=self._matrix_init,
        trainable=True)

    if self._with_bias:
      self._matrix_bias = self.add_weight(
          name='matrix_bias',
          shape=(1, self._dim_m, self._dim_m),
          initializer='uniform',
          trainable=True)

    super(MLayer, self).build(input_shape)

  def call(self, x):
    if not self._with_bias:
      mat = tf.einsum('amn,...a->...mn', self._rep_to_exp_tensor, x)
    else:
      mat = tf.einsum('amn,...a->...mn', self._rep_to_exp_tensor,
                      x) + self._matrix_bias
    if self._matrix_squarings_exp is None:
      return tf.linalg.expm(mat)
    # Approximation of exp(mat) as (1+mat/k)**k with k = 2**MATRIX_SQUARINGS_EXP
    mat = mat * 0.5**self._matrix_squarings_exp + tf.eye(self._dim_m)
    for _ in range(self._matrix_squarings_exp):
      mat = tf.einsum('...ij,...jk->...ik', mat, mat)
    return mat

  def compute_output_shape(self, input_shape):
    return input_shape[0], self._dim_m, self._dim_m

  def get_config(self):
    config = dict(super().get_config())
    config['dim_m'] = self._dim_m
    config['matrix_init'] = self._matrix_init
    config['with_bias'] = self._with_bias
    config['matrix_squarings_exp'] = self._matrix_squarings_exp
    return config
