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

"""Utilities for Learner architectures."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import gin
import numpy as np
import py_utils
import sonnet as snt
import tensorflow.compat.v1 as tf
import tf_utils

nest = tf.contrib.framework.nest

RollingFeaturesState = collections.namedtuple("RollingFeaturesState",
                                              ["ms", "rms"])


@gin.configurable
class RollingFeatures(snt.AbstractModule):
  """Helper to construct different decay, momentum, and rms rolling averages.

  These are used as features in a learned optimizer and are the exact same as
  done in
  in SGD+Momentum and RMSProp.

  Unlike Adam / RMSProp, we accumulate values at multiple decay rates.
  """

  def __init__(self,
               name="RollingFeatures",
               include_rms=False,
               decays=None,
               num_features=30,
               min=0.0001,  # pylint: disable=redefined-builtin
               max=0.5,  # pylint: disable=redefined-builtin
               **kwargs):
    self.include_rms = include_rms
    if decays is None:
      self.decays = tf.constant(
          1 - np.logspace(np.log10(min), np.log10(max), num_features),
          dtype=tf.float32)
    else:
      self.decays = tf.constant(decays)

    super(RollingFeatures, self).__init__(name=name, **kwargs)
    self()
    self.var_shapes = []

  def _build(self):
    pass

  @snt.reuse_variables
  def initial_state(self, shapes):
    ms = []
    rms = []
    self.var_shapes = shapes
    for s in shapes:
      # the optimizer works on batches of data, thus 1 batch size.
      n_dims = int(np.prod(s))

      ms.append(tf.zeros([n_dims, self.decays.shape.as_list()[0]]))
      rms.append(tf.zeros([n_dims, self.decays.shape.as_list()[0]]))

    return RollingFeaturesState(ms=ms, rms=rms)

  @snt.reuse_variables
  def current_state(self, shapes):
    init_state = self.initial_state(shapes)
    return tf_utils.make_variables_matching(init_state)

  @snt.reuse_variables
  def next_state(self, state, grads):
    pad_decay = tf.expand_dims(self.decays, 0)
    new_ms_list, new_rms_list = [], []
    for ms, rms, g, var_shape in py_utils.eqzip(state.ms, state.rms, grads,
                                                self.var_shapes):

      def single_update(grad, ms, rms):
        grad = tf.reshape(grad, [-1, 1])
        new_ms = ms * pad_decay + grad * (1 - pad_decay)
        if self.include_rms:
          new_rms = rms * pad_decay + tf.square(grad) * (1 - pad_decay)
          return new_ms, new_rms
        else:
          return new_ms, rms

      if isinstance(g, tf.IndexedSlices):
        # pylint: disable=unbalanced-tuple-unpacking
        new_ms, new_rms = indexed_slices_apply_dense2(single_update, var_shape,
                                                      g, [ms, rms], 2)
      else:
        new_ms, new_rms = single_update(g, ms, rms)

      new_ms_list.append(new_ms)
      new_rms_list.append(new_rms)
    return RollingFeaturesState(ms=new_ms_list, rms=new_rms_list)


def indexed_slices_apply_dense2(fn, var_shape, g_inp, dense_var_inp, n_outs):
  """Helper function to work with sparse tensors.

  dense_var_inp has the leading 2 dimensions collapsed forming shape [n_words *
  n_words_feat, n_feat]
  g_inp on the otherhand is [n_words, n_words_feat]
  var_shape is static and is [n_words, n_words_feat]

  Arguments:
    fn: (gradient: tf.Tensor, *var_args: tf.Tensor) -> [tf.Tensor]
    var_shape: list
    g_inp: tf.IndexedSlices
    dense_var_inp: tf.Tensor list.
    n_outs: int
  Returns:
    dense outputs
  """
  grad_idx, grad_value = accumulate_sparse_gradients(g_inp)

  n_words, n_word_feat = var_shape

  args = []
  for a_possibly_nest in dense_var_inp:

    def do_on_tensor(a):
      n_feat = a.shape.as_list()[1]
      n_active = tf.size(grad_idx)
      reshaped = tf.reshape(a, [n_words, n_word_feat, n_feat])
      sub_reshaped = tf.gather(reshaped, grad_idx)
      return tf.reshape(sub_reshaped, [n_active * n_word_feat, n_feat])

    args.append(nest.map_structure(do_on_tensor, a_possibly_nest))

  returns = fn(grad_value, *args)

  def undo((full_val, sub_val)):
    """Undo the slices."""
    if tf.shape(full_val).shape.as_list()[0] != 2:
      raise NotImplementedError(
          "TODO(lmetz) other than this is not implemented.")
    n_words, n_word_feat = var_shape
    _, n_feat = sub_val.shape.as_list()
    n_active = tf.size(grad_idx)

    shape = [n_active, n_word_feat * n_feat]
    in_shape_form = tf.reshape(sub_val, shape)

    new_shape = [n_words, n_word_feat * n_feat]
    mask_shape = [n_words, n_word_feat * n_feat]

    scattered = tf.scatter_nd(
        tf.reshape(tf.to_int32(grad_idx), [-1, 1]),
        in_shape_form,
        shape=new_shape)
    mask = tf.scatter_nd(
        tf.reshape(tf.to_int32(grad_idx), [-1, 1]),
        tf.ones_like(in_shape_form),
        shape=mask_shape)

    # put back into the flat format
    scattered = tf.reshape(scattered, [n_words * n_word_feat, n_feat])
    mask = tf.reshape(mask, [n_words * n_word_feat, n_feat])

    # this is the update part / fake scatter_update but with gradients.
    return full_val * (1 - mask) + scattered * mask

  dense_outs = []
  for ret, dense_v in list(py_utils.eqzip(returns, dense_var_inp[0:n_outs])):
    flat_out = map(undo,
                   py_utils.eqzip(nest.flatten(dense_v), nest.flatten(ret)))
    dense_outs.append(nest.pack_sequence_as(dense_v, flat_out))
  return dense_outs


def accumulate_sparse_gradients(grad):
  """Accumulates repeated indices of a sparse gradient update.

  Args:
    grad: a tf.IndexedSlices gradient

  Returns:
    grad_indices: unique indices
    grad_values: gradient values corresponding to the indices
  """

  grad_indices, grad_segments = tf.unique(grad.indices)
  grad_values = tf.unsorted_segment_sum(grad.values, grad_segments,
                                        tf.size(grad_indices))
  return grad_indices, grad_values


def tanh_embedding(x):
  """Embed time in a format usable by a neural network.

  This embedding involves dividing x by different timescales and running through
  a squashing function.
  Args:
    x: tf.Tensor
  Returns:
    tf.Tensor
  """
  mix_proj = []
  for i in [3, 10, 30, 100, 300, 1000, 3000, 10000, 300000]:
    mix_proj.append(tf.tanh(tf.to_float(tf.to_float(x) / float(i)) - 1.))
  return tf.stack(mix_proj)


class SecondMomentNormalizer(snt.AbstractModule):

  def _build(self, x, is_training=True):
    normed = x * tf.rsqrt(1e-5 +
                          tf.reduce_mean(tf.square(x), axis=0, keep_dims=True))
    return normed
