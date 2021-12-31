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

"""Keras Layers for differentiable local sequence alignment."""

from typing import Optional, Sequence, Tuple, Type, Union

import gin
import tensorflow as tf

from dedal import pairs as pairs_lib
from dedal import smith_waterman
from dedal.models import initializers


Initializer = tf.initializers.Initializer
LayerFactory = Type[tf.keras.layers.Layer]
SWParams = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
# Score, path and Smith-Waterman params
AlignmentOutput = Tuple[tf.Tensor, Optional[tf.Tensor], SWParams]
NaiveAlignmentOutput = Tuple[tf.Tensor, tf.Tensor, SWParams]


@gin.configurable
class PairwiseBilinearDense(tf.keras.layers.Layer):
  """Computes (learnable) bilinear form for (batched) sets of vector pairs."""

  def __init__(
      self,
      use_kernel = True,
      use_bias = True,
      trainable_kernel = True,
      trainable_bias = True,
      kernel_init = initializers.SymmetricKernelInitializer(),
      bias_init = tf.initializers.Zeros(),
      symmetric_kernel = True,
      dropout = 0.0,
      symmetric_dropout = True,
      sqrt_norm = True,
      activation=None,
      mask_penalty = -1e9,
      **kwargs):
    super().__init__(**kwargs)
    self._use_kernel = use_kernel
    self._use_bias = use_bias
    self._trainable_kernel = trainable_kernel
    self._trainable_bias = trainable_bias
    self._kernel_init = kernel_init
    self._bias_init = bias_init
    self._symmetric_kernel = symmetric_kernel
    self._dropout = dropout
    self._symmetric_dropout = symmetric_dropout
    self._sqrt_norm = sqrt_norm
    self._activation = activation
    self._mask_penalty = mask_penalty

  def build(self, input_shape):
    if self._use_kernel:
      emb_dim = input_shape[-1]
      self.kernel = self.add_weight(
          shape=(emb_dim, emb_dim),
          initializer=self._kernel_init,
          trainable=self._trainable_kernel,
          name='bilinear_form_kernel')
    if self._use_bias:
      self.bias = self.add_weight(
          shape=(),
          initializer=self._bias_init,
          trainable=self._trainable_bias,
          name='bilinear_form_bias')
    noise_shape = None
    if self._symmetric_dropout:
      noise_shape = [input_shape[0]] + [1] + input_shape[2:]
    self.dropout = tf.keras.layers.Dropout(
        rate=self._dropout, noise_shape=noise_shape)

  def call(self, inputs, mask=None, training=None):
    """Evaluates bilinear form for (batched) sets of vector pairs.

    Args:
      inputs: a tf.Tensor<float>[batch, 2, len, dim] representing two inputs.
      mask: a tf.Tensor<float>[batch, 2, len] to account for padding.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      A tf.Tensor<float>[batch, len, len] s.t.
        out[n][i][j] := activation( (x[n][i]^{T} W y[n][j]) / norm_factor + b),
      where the bilinear form matrix W can optionally be set to be the identity
      matrix (use_kernel = False) or optionally frozen to its initialization
      value (trainable_kernel = False) and the scalar bias b can be optionally
      set to zero (use_bias = False) or likewise optionally frozen to its
      initialization value (trainable_bias=False). If sqrt_norm is True, the
      scalar norm_factor above is set to sqrt(d), following dot-product
      attention. Otherwise, norm_factor = 1.0.
      Finally, if either masks_x[n][i] = 0 or masks_y[n][j] = 0 and mask_penalty
      is not None, then
        out[n][i][j] = mask_penalty
      instead.
    """
    inputs = self.dropout(inputs, training=training)
    x, y = inputs[:, 0], inputs[:, 1]
    if not self._use_kernel:
      output = tf.einsum('ijk,ilk->ijl', x, y)
    else:
      w = self.kernel
      if self._symmetric_kernel:
        w = 0.5 * (w + tf.transpose(w))
      output = tf.einsum('nir,rs,njs->nij', x, w, y)
    if self._sqrt_norm:
      dim_x, dim_y = tf.shape(x)[-1], tf.shape(y)[-1]
      dim = tf.sqrt(tf.cast(dim_x * dim_y, output.dtype))
      output /= tf.sqrt(dim)
    if self._use_bias:
      output += self.bias
    if self._activation is not None:
      output = self._activation(output)
    if self._mask_penalty is not None and mask is not None:
      paired_masks = pairs_lib.pair_masks(mask[:, 0], mask[:, 1])
      output = tf.where(paired_masks, output, self._mask_penalty)

    return output


@gin.configurable
class SoftSymmetricAlignment(tf.keras.Model):
  """Implements the soft symmetric alignment layer in Bepler et al. 2019."""

  def __init__(
      self,
      emb_dim = 128,
      norm = 'l2',
      proj = True,
      batch_size = None,
      kernel_init = tf.initializers.HeUniform(),
      bias_init = tf.initializers.Zeros(),
      return_att_weights = False,
      **kwargs):
    super().__init__(**kwargs)
    self._norm = norm
    self._proj = proj
    self._batch_size = batch_size
    self._return_att_weights = return_att_weights

    if self._norm not in ('l1', 'l2'):
      raise ValueError(f'Option {self._norm} not recognized.')

    self.dense = tf.keras.layers.Dense(
        emb_dim,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
        name='linear_projection')
    self.softmax_a = tf.keras.layers.Softmax(axis=2)
    self.softmax_b = tf.keras.layers.Softmax(axis=1)

  def call(
      self,
      embeddings,
      mask=None):
    """Computes the forward pass for the soft symmetric alignment layer.

    Args:
      embeddings: A tf.Tensor[batch, 2, len, dim] with the embeddings of the
        two sequences.
      mask: A tf.Tensor[batch, 2, len] with the paddings masks of the two
        sequences.

    Returns:
      The soft symmetric alignment similarity scores, as defined by the paper
        Bepler et al. - Learning protein sequence embeddings using information
        from structure. ICLR 2019,
      represented by a 1D tf.Tensor of dimension batch_size.
      If return_att_weights is True, it will additionally return the soft
      symmetric alignments weights as a tf.Tensor<float>[batch, len, len] with
      entries in [0, 1].
    """
    if self._proj:
      embeddings = self.dense(embeddings)

    pair_dist = self.pairwise_distance(embeddings[:, 0], embeddings[:, 1])
    pair_mask = pairs_lib.pair_masks(mask[:, 0], mask[:, 1])

    a = self.softmax_a(-pair_dist, pair_mask)
    b = self.softmax_b(-pair_dist, pair_mask)
    att_weights = tf.where(pair_mask, (a + b - a * b), 0.0)
    scores = -tf.reduce_sum(att_weights * pair_dist, (1, 2))
    scores /= tf.reduce_sum(att_weights, (1, 2))
    return (scores, att_weights) if self._return_att_weights else scores

  def pairwise_distance(
      self,
      x,
      y,
      low_mem = True,
  ):
    if self._norm == 'l1':
      if low_mem:
        # Avoid vectorization across batch dimension to reduce memory usage
        batch_size = tf.shape(x)[0]  # equals tf.shape(y)[0]
        dtype = x.dtype  # equals y.dtype

        i0 = tf.constant(0)
        pair_dist0 = tf.TensorArray(dtype=dtype, size=batch_size)

        def cond(i, _):
          return i < batch_size

        def body(i, pair_dist):
          x_i, y_i = x[i], y[i]
          abs_diff_i = tf.abs(tf.expand_dims(x_i, 1) - tf.expand_dims(y_i, 0))
          pair_dist_i = tf.reduce_sum(abs_diff_i, -1)
          return i + 1, pair_dist.write(i, pair_dist_i)

        _, pair_dist = tf.while_loop(cond, body, (i0, pair_dist0),
                                     maximum_iterations=self._batch_size)
        pair_dist = pair_dist.stack()
      # Note: this is quite memory-intensive. One could trade-off storage vs
      # runtime complexity by e.g. using scan along the trailing axis or along
      # the leading axis, as done above
      else:
        abs_diff = tf.abs(tf.expand_dims(x, 2) - tf.expand_dims(y, 1))
        pair_dist = tf.reduce_sum(abs_diff, -1)
    elif self._norm == 'l2':
      x_normsq = tf.reduce_sum(x ** 2, -1)
      y_normsq = tf.reduce_sum(y ** 2, -1)
      x_dot_y = tf.einsum('ijk,ilk->ijl', x, y)
      pair_dist = x_normsq[:, :, None] + y_normsq[:, None, :] - 2.0 * x_dot_y
    else:
      raise ValueError(f'Option {self._norm} not recognized.')
    return pair_dist


@gin.configurable
class ConstantGapPenalties(tf.keras.layers.Layer):
  """Wraps position-independent gap penalty parameters for differentiable SW."""

  def __init__(self,
               trainable_gap_open = True,
               trainable_gap_extend = True,
               gap_open_init = tf.initializers.Constant(12.0),
               gap_extend_init = tf.initializers.Constant(1.0),
               **kwargs):
    super().__init__(**kwargs)
    self._trainable_gap_open = trainable_gap_open
    self._trainable_gap_extend = trainable_gap_extend
    self._gap_open_init = gap_open_init
    self._gap_extend_init = gap_extend_init

  def build(self, _):
    self._gap_open = self.add_weight(
        shape=(),
        initializer=self._gap_open_init,
        constraint=tf.keras.constraints.NonNeg(),
        trainable=self._trainable_gap_open,
        name='gap_open')
    self._gap_extend = self.add_weight(
        shape=(),
        initializer=self._gap_extend_init,
        constraint=tf.keras.constraints.NonNeg(),
        trainable=self._trainable_gap_extend,
        name='gap_extend')

  def call(
      self, embeddings, mask=None):
    """Computes pos. indepedent gap open and gap extend params from embeddings.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] representing the
        embeddings of the two inputs.
      mask: a tf.Tensor<float>[batch, 2, len] representing the padding masks of
        the two inputs.

    Returns:
      A 2-tuple (gap_open, gap_extend) of tf.Tensor<float>[batch, len, len].
    """
    batch_size = tf.shape(embeddings)[0]
    tile = lambda t: t * tf.ones(batch_size, dtype=t.dtype)
    return tile(self._gap_open), tile(self._gap_extend)


@gin.configurable
class ConstantSharedGapPenalties(tf.keras.layers.Layer):
  """Wraps tied position-independent gap penalty param for differentiable SW."""

  def __init__(self,
               trainable_gap_penalty = True,
               gap_penalty_init = tf.initializers.Constant(11.0),
               **kwargs):
    super().__init__(**kwargs)
    self._trainable_gap_penalty = trainable_gap_penalty
    self._gap_penalty_init = gap_penalty_init

  def build(self, _):
    self._gap_penalty = self.add_weight(
        shape=(),
        initializer=self._gap_penalty_init,
        constraint=tf.keras.constraints.NonNeg(),
        trainable=self._trainable_gap_penalty,
        name='gap_penalty')

  def call(
      self, embeddings, mask=None):
    """Computes tied pos. independent gap open / extend params from embeddings.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] representing the
        embeddings of the two inputs.
      mask: a tf.Tensor<float>[batch, 2, len] representing the padding masks of
        the two inputs.

    Returns:
      A 2-tuple (gap_open, gap_extend) of tf.Tensor<float>[batch, len, len],
      where gap_open = gap_extend (linear gap penalty model).
    """
    batch_size = tf.shape(embeddings)[0]
    tile = lambda t: t * tf.ones(batch_size, dtype=t.dtype)
    return tile(self._gap_penalty), tile(self._gap_penalty)


@gin.configurable
class ContextualGapPenalties(tf.keras.Model):
  """Wraps untied contextual gap penalty parameters for differentiable SW.

  Gap open and gap extend penalties will be computed without parameter sharing.
  """

  def __init__(self,
               gap_open_cls = gin.REQUIRED,
               gap_extend_cls = gin.REQUIRED,
               **kwargs):
    super().__init__(**kwargs)
    self._gap_open = gap_open_cls()
    self._gap_extend = gap_extend_cls()

  def call(self,
           embeddings,
           mask = None,
           training = None):
    """Computes contextual gap open and gap extend params from embeddings.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] representing the
        embeddings of the two inputs.
      mask: a tf.Tensor<float>[batch, 2, len] representing the padding masks of
        the two inputs.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      A 2-tuple (gap_open, gap_extend) of tf.Tensor<float>[batch, len, len].
    """
    return (self._gap_open(embeddings, mask=mask, training=training),
            self._gap_extend(embeddings, mask=mask, training=training))


@gin.configurable
class ContextualSharedGapPenalties(tf.keras.layers.Layer):
  """Wraps shared contextual gap penalty parameters for differentiable SW.

  Unlike for ContextualGapPenalties, the gap open and gap extend penalties will
  differ only by a learnable constant satisfying gap open - gap extend >= 0.
  """

  def __init__(self,
               gap_cls = gin.REQUIRED,
               gap_open_bias_init = tf.initializers.Constant(11.0),
               gap_open_bias_trainable = True,
               **kwargs):
    super().__init__(**kwargs)
    self._gap = gap_cls()
    self._gap_open_bias = self.add_weight(
        shape=(),
        initializer=gap_open_bias_init,
        constraint=tf.keras.constraints.NonNeg(),
        trainable=gap_open_bias_trainable,
        name='gap_open_bias')

  def call(self,
           embeddings,
           mask = None,
           training = None):
    """Computes contextual gap open and gap extend params from embeddings.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] representing the
        embeddings of the two inputs.
      mask: a tf.Tensor<float>[batch, 2, len] representing the padding masks of
        the two inputs.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      A 2-tuple (gap_open, gap_extend) of tf.Tensor<float>[batch, len, len].
    """
    gap_pen = self._gap(embeddings, mask=mask, training=training)
    return gap_pen + self._gap_open_bias, gap_pen


@gin.configurable
class SoftAligner(tf.keras.Model):
  """Computes soft Smith-Waterman scores via regularization."""

  def __init__(self,
               similarity_cls = PairwiseBilinearDense,
               gap_pen_cls = ContextualGapPenalties,
               align_fn=smith_waterman.perturbed_alignment_score,
               eval_align_fn=smith_waterman.unperturbed_alignment_score,
               trainable = True,
               **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self._similarity = similarity_cls()
    self._gap_pen = gap_pen_cls()
    self._align_fn = align_fn
    self._eval_align_fn = align_fn if eval_align_fn is None else eval_align_fn

  def call(self, embeddings, mask=None, training=None):
    """Computes soft Smith-Waterman scores via regularization.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] containing pairs of
        sequence embeddings (with the sequence lengths).
      mask: An optional token mask to account for padding.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      An AlignmentOutput which is a 3-tuple made of:
        - The alignment scores: tf.Tensor<float>[batch].
        - If provided by the alignment function, the alignment matrix as a
          tf.Tensor<int>[batch, len, len, 9]. Otherwise None.
        - A 3-tuple containing the Smith-Waterman parameters: similarities, gap
          open and gap extend. Similaries is tf.Tensor<float>[batch, len, len],
          the gap penalties can be either tf.Tensor<float>[batch] or
          tf.Tensor<float>[batch, len, len].
    """
    sim_mat = self._similarity(embeddings, mask=mask, training=training)
    gap_open, gap_extend = self._gap_pen(
        embeddings, mask=mask, training=training)
    sw_params = (sim_mat, gap_open, gap_extend)
    results = (self._align_fn if training else self._eval_align_fn)(*sw_params)
    results = (results,) if not isinstance(results, Sequence) else results
    # TODO(oliviert): maybe inject some metrics here.
    return (results + (None,))[:2] + (sw_params,)


@gin.configurable
class NaiveAligner(tf.keras.Model):
  """Computes aligments and scores as a set of binary classifications."""

  def __init__(self,
               similarity_cls = PairwiseBilinearDense,
               trainable = True,
               **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self._similarity = similarity_cls()

  def call(self, embeddings, mask=None, training=None):
    r"""Computes aligments and scores as a set of binary classifications.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] containing pairs of
        sequence embeddings (with the sequence lengths).
      mask: An optional token mask to account for padding.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      A NaiveAlignmentOutput which is a 3-tuple made of:
        - The alignment scores: tf.Tensor<float>[batch]. In particular, this
          layer defines the score of an alignment as
            \sum_{i, j} similarities[i, j] sigmoid(similarities[i, j]).
        - The pairwise match probabilities: tf.Tensor<int>[batch, len, len].
        - A 3-tuple containing the similarities, gap open and gap extend
          penalties. Similaries is tf.Tensor<float>[batch, len, len] and equals
          the logits of the pairwise match probabilities. The gap penalties are
          tf.Tensor<float>[batch] of zeroes, present for consistency in the
          output signature.
    """
    batch, dtype = tf.shape(embeddings)[0], embeddings.dtype
    sim_mat = self._similarity(embeddings, mask=mask, training=training)
    match_indicators_pred = tf.nn.sigmoid(sim_mat)
    scores = tf.reduce_sum(sim_mat * match_indicators_pred, axis=[1, 2])
    sw_params = (sim_mat, tf.zeros([batch], dtype), tf.zeros([batch], dtype))
    return scores, match_indicators_pred, sw_params


@gin.configurable
class SSAAligner(tf.keras.Model):
  """Computes aligments and scores ala Bepler et al. ICLR 2019."""

  def __init__(self,
               similarity_cls = SoftSymmetricAlignment,
               trainable = True,
               **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self._similarity = similarity_cls()

  def call(self, embeddings, mask=None, training=None):
    """Computes aligments and scores ala Bepler et al. ICLR 2019.

    Args:
      embeddings: a tf.Tensor<float>[batch, 2, len, dim] containing pairs of
        sequence embeddings (with the sequence lengths).
      mask: An optional token mask to account for padding.
      training: whether to run the layer for train (True), eval (False) or let
        the Keras backend decide (None).

    Returns:
      A NaiveAlignmentOutput which is a 3-tuple made of:
        - The alignment scores: tf.Tensor<float>[batch].
        - The pairwise match probabilities: tf.Tensor<int>[batch, len, len].
        - A 3-tuple containing the similarities, gap open and gap extend. Here
          similaries is tf.Tensor<float>[batch, len, len] that simply encodes
          the padding mask, taking value 0.0 for "real" tokens or 1e9 for
          padding / special tokens. The gap penalties are
          tf.Tensor<float>[batch] of zeroes, present for consistency in the
          output signature.
    """
    batch, dtype = tf.shape(embeddings)[0], embeddings.dtype
    scores, match_indicators_pred = self._similarity(embeddings, mask=mask)
    # Here sim_mat has no real purpose other than passing the padding mask to
    # the loss and metrics for the corresponding output head.
    sim_mat = tf.where(pairs_lib.pair_masks(mask[:, 0], mask[:, 1]), 0.0, 1e9)
    sw_params = (sim_mat, tf.zeros([batch], dtype), tf.zeros([batch], dtype))
    return scores, match_indicators_pred, sw_params
