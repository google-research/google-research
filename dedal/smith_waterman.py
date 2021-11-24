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

"""Perturbed Smith-Waterman."""

import functools
from typing import Tuple, Union

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from dedal import alignment


_GUMBEL = 'gumbel'
_NORMAL = 'normal'
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)


def wavefrontify(tensor):
  """Rearranges input tensor for vectorized wavefront algorithm.

  Args:
    tensor: A tf.Tensor<dtype>[batch, len1, len2, s], where the first and last
     dimensions will be effectively treated as batch dimensions.

  Returns:
    A single tf.Tensor<dtype>[len1 + len2 - 1, s, len1, batch] satisfying
      out[k][a][i][n] = t[n][i][k - i][a]
    if the RHS is well-defined, and 0 otherwise.
    In other words, for each len1 x len2 matrix t[n, :, :, a], out[:, a, :, n]
    is a (len1 + len2 - 1) x len1 matrix whose rows correspond to antidiagonals
    of t[n, :, :, a].
  """
  b = tf.shape(tensor)[0]
  # l1, l2 = tf.shape(tensor)[1], tf.shape(tensor)[2]
  l1, l2 = tensor.shape[1], tensor.shape[2]
  s = tf.shape(tensor)[3]
  n_pad, padded_len = l1 - 1, l1 + l2 - 1

  ta = tf.TensorArray(tensor.dtype, size=l1, clear_after_read=True)
  for i in range(l1):
    row_i = tf.squeeze(tf.slice(tensor, [0, i, 0, 0], [b, 1, l2, s]), axis=1)
    row_i = tf.pad(row_i, [[0, 0], [n_pad, n_pad], [0, 0]])
    row_i = tf.slice(row_i, [0, n_pad - i, 0], [b, padded_len, s])
    ta = ta.write(i, row_i)  # row_i[b, padded_len, s]
  ta = ta.stack()  # ta[l1, b, padded_len, s]

  return tf.transpose(ta, (2, 3, 0, 1))  # out[padded_len, s, l1, b]


def hard_sw_affine(
    weights,
    tol = 1e-6,
):
  """Solves the Smith-Waterman LP, computing both optimal scores and alignments.

  Args:
    weights: A tf.Tensor<float>[batch, len1, len2, 9] (len1 <= len2) of edge
      weights (see function weights_from_sim_mat above for an in-depth
      description).
    tol: A small positive constant to ensure the first transition begins at the
      start state. Note(fllinares): this might not be needed anymore, test!

  Returns:
    Two tensors corresponding to the scores and alignments, respectively.
    + The first tf.Tensor<float>[batch] contains the Smith-Waterman scores for
      each pair of sequences in the batch.
    + The second tf.Tensor<int>[batch, len1, len2, 9] contains binary entries
      indicating the trajectory of the indices along the optimal path for each
      sequence pair, by having a one along the taken edges, with nine possible
      edges for each i,j.
  """
  # Gathers shape and type variables.
  b, l1, l2 = tf.shape(weights)[0], weights.shape[1], weights.shape[2]
  padded_len = l1 + l2 - 1
  dtype = weights.dtype
  inf = alignment.large_compatible_positive(dtype)

  # Rearranges input tensor for vectorized wavefront iterations.
  weights = wavefrontify(weights)  # [padded_len, s, l1, b]
  w_m, w_x, w_y = tf.split(weights, [4, 2, 3], axis=1)

  ### FORWARD

  # Auxiliary functions + syntatic sugar.
  def slice_lead_dims(
      t,
      k,
      s,
  ):
    """Returns t[k][:s] for "wavefrontified" tensors."""
    return tf.squeeze(tf.slice(t, [k, 0, 0, 0], [1, s, l1, b]), 0)

  # "Wavefrontified" tensors contain invalid entries that need to be masked.
  def slice_inv_mask(k):
    """Masks invalid and sentinel entries in wavefrontified tensors."""
    j_range = k - tf.range(1, l1 + 1, dtype=tf.int32) + 2
    return tf.logical_and(j_range > 0, j_range <= l2)  # True iff valid.

  # Setups reduction operators.
  def reduce_max_with_argmax(
      t, axis = 0):
    # Note(fllinares): I haven't yet managed to beat the performance of this
    # (wasteful) implementation with tf.argmax + tf.gather / tf.gather_nd :(
    t_max = tf.reduce_max(t, axis=axis)
    t_argmax = tf.argmax(t, axis=axis, output_type=tf.int32)
    return t_max, t_argmax

  # Initializes forward recursion.
  v_p2, v_p1 = tf.fill([3, l1, b], -inf), tf.fill([3, l1, b], -inf)
  # Ensures that edges cases for which all substitution costs are negative
  # result in a score of zero and an empty alignment.
  v_opt = tf.zeros(b, dtype=dtype)
  k_opt, i_opt = -tf.ones(b, dtype=tf.int32), -tf.ones(b, dtype=tf.int32)
  d_all = tf.TensorArray(tf.int32, size=padded_len, clear_after_read=True)

  # Runs forward Smith-Waterman recursion.
  for k in range(padded_len):
    # NOTE(fllinares): shape information along the batch dimension seems to get
    # lost in the edge-case b=1
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(v_p2, tf.TensorShape([3, None, None])),
                          (v_p1, tf.TensorShape([3, None, None])),
                          (v_opt, tf.TensorShape([None,])),
                          (k_opt, tf.TensorShape([None,])),
                          (i_opt, tf.TensorShape([None,]))])
    # inv_mask: masks out invalid entries for v_p2, v_p1 and v_opt updates.
    inv_mask_k = slice_inv_mask(k)[tf.newaxis, :, tf.newaxis]

    o_m = slice_lead_dims(w_m, k, 4) + alignment.top_pad(v_p2, tol)
    o_x = slice_lead_dims(w_x, k, 2) + v_p1[:2]
    v_p1 = alignment.left_pad(v_p1[:, :-1], -inf)
    o_y = slice_lead_dims(w_y, k, 3)  + v_p1

    v_m, d_m = reduce_max_with_argmax(o_m, axis=0)
    v_x, d_x = reduce_max_with_argmax(o_x, axis=0)
    v_y, d_y = reduce_max_with_argmax(o_y, axis=0)
    v = tf.where(inv_mask_k, tf.stack([v_m, v_x, v_y]), -inf)
    d = tf.stack([d_m, d_x + 1, d_y + 1])  # Accounts for start state (0).

    v_p2, v_p1 = v_p1, v
    v_opt_k, i_opt_k = reduce_max_with_argmax(v[0], axis=0)
    update_cond = v_opt_k > v_opt
    v_opt = tf.where(update_cond, v_opt_k, v_opt)
    k_opt = tf.where(update_cond, k, k_opt)
    i_opt = tf.where(update_cond, i_opt_k, i_opt)
    d_all = d_all.write(k, d)

  ### BACKTRACKING

  # Creates auxiliary tensors to encode backtracking "actions".
  steps_k = tf.convert_to_tensor([0, -2, -1, -1], dtype=tf.int32)
  steps_i = tf.convert_to_tensor([0, -1, 0, -1], dtype=tf.int32)
  trans_enc = tf.constant([[10, 10, 10, 10],
                           [1, 2, 3, 4],
                           [10, 5, 6, 10],
                           [10, 7, 8, 9]], dtype=tf.int32)  # [m_curr, m_prev]
  samp_idx = tf.range(b, dtype=tf.int32)

  # Initializes additional backtracking variables.
  m_opt = tf.ones(b, dtype=tf.int32)  # Init at match states (by definition).
  paths_sp = tf.TensorArray(tf.int32, size=padded_len, clear_after_read=True)

  # Runs Smith-Waterman backtracking.
  for k in range(padded_len - 1, -1, -1):
    # NOTE(fllinares): shape information along the batch dimension seems to get
    # lost in the edge-case b=1
    tf.autograph.experimental.set_loop_options(
        shape_invariants=[(m_opt, tf.TensorShape([None,]))])
    # Computes tentative next indices for each alignment.
    k_opt_n = k_opt + tf.gather(steps_k, m_opt)
    i_opt_n = i_opt + tf.gather(steps_i, m_opt)
    # Computes tentative next state types for each alignment.
    m_opt_n_idx = tf.stack(
        [tf.maximum(m_opt - 1, 0), tf.maximum(i_opt, 0), samp_idx], -1)
    m_opt_n = tf.gather_nd(d_all.read(k), m_opt_n_idx)
    # Computes tentative next sparse updates for paths tensor.
    edges_n = tf.gather_nd(trans_enc, tf.stack([m_opt, m_opt_n], -1))
    paths_sp_n = tf.stack([samp_idx, i_opt + 1, k_opt - i_opt + 1, edges_n], -1)

    # Indicates alignments to be updated in this iteration.
    cond = tf.logical_and(k_opt == k, m_opt != 0)
    # Conditionally applies updates for each alignment.
    k_opt = tf.where(cond, k_opt_n, k_opt)
    i_opt = tf.where(cond, i_opt_n, i_opt)
    m_opt = tf.where(cond, m_opt_n, m_opt)
    paths_sp_k = tf.where(cond[:, None], paths_sp_n, tf.zeros([b, 4], tf.int32))
    paths_sp = paths_sp.write(k, paths_sp_k)  # [0, 0, 0, 0] used as dummy upd.

  # Applies sparse updates, building paths tensor.
  paths_sp = tf.reshape(paths_sp.stack(), [-1, 4])  # [(padded_len * b), 4]
  paths_sp_idx, paths_sp_upd = paths_sp[:, :3], paths_sp[:, 3]
  paths = tf.scatter_nd(paths_sp_idx, paths_sp_upd, [b, l1 + 1, l2 + 1])
  paths = paths[:, 1:, 1:]  # Removes sentinel row/col.
  # Represents paths tensor using one-hot encoding over 9 states.
  paths = tf.one_hot(paths, tf.reduce_max(trans_enc))[:, :, :, 1:]
  return v_opt, paths


@gin.configurable
def unperturbed_alignment_score(sim_mat,
                                gap_open,
                                gap_extend):
  """Noiseless alignment score and paths from Smith-Waterman parameters."""
  sw_params = alignment.weights_from_sim_mat(sim_mat, gap_open, gap_extend)
  return hard_sw_affine(sw_params)


def sample_noise_with_gradients(
    noise, shape):
  """Samples a noise tensor according to a distribution with its gradient.

  Args:
   noise: (str) a type of supported noise distribution.
   shape: tf.Tensor<int>, the shape of the tensor to sample.

  Returns:
   A tuple Tensor<float>[shape], Tensor<float>[shape] that corresponds to the
   sampled noise and the gradient of log the underlying probability
   distribution function. For instance, for a gaussian noise (normal), the
   gradient is equal to the noise itself.

  Raises:
   ValueError in case the requested noise distribution is not supported.
   See SUPPORTED_NOISES for the list of supported distributions.
  """
  if noise not in SUPPORTED_NOISES:
    raise ValueError('{} noise is not supported. Use one of [{}]'.format(
        noise, SUPPORTED_NOISES))

  if noise == _GUMBEL:
    sampler = tfp.distributions.Gumbel(0.0, 1.0)
    samples = sampler.sample(shape)
    gradients = 1 - tf.math.exp(-samples)
  elif noise == _NORMAL:
    sampler = tfp.distributions.Normal(0.0, 1.0)
    samples = sampler.sample(shape)
    gradients = samples

  return samples, gradients


@gin.configurable
def perturbed_alignment_score(
    sim_mat,
    gap_open,
    gap_extend,
    noise = 'normal',
    sigma = 0.1,
    num_samples = 1,
    stop_paths_gradient = True):
  """Perturbed alignment score and paths from Smith-Waterman parameters."""
  noise_sampler = functools.partial(sample_noise_with_gradients, noise)

  @tf.custom_gradient
  def forward(sw_params):
    # Perturbs the Smith-Waterman LP parameters.
    sim_mat, gap_open, gap_extend = sw_params[0], sw_params[1], sw_params[2]
    sw_params = alignment.weights_from_sim_mat(sim_mat, gap_open, gap_extend)
    shape, dtype = tf.shape(sw_params), sw_params.dtype
    pert_shape = tf.concat([[num_samples], shape], axis=0)
    noise, noise_grad = tf.nest.map_structure(
        lambda t: tf.cast(t, dtype), noise_sampler(pert_shape))
    pert_sw_params = tf.expand_dims(sw_params, 0) + sigma * noise
    pert_sw_params = tf.reshape(pert_sw_params,
                                tf.concat([[-1], shape[1:]], axis=0))
    # Computes optimal Smith-Waterman scores and alignments.
    scores, paths = hard_sw_affine(pert_sw_params)

    # Average scores and outputs over random perturbations
    def perturbation_reshape(t):
      new_shape = tf.concat([[num_samples], [-1], tf.shape(t)[1:]], 0)
      return tf.reshape(t, new_shape)

    scores = perturbation_reshape(scores)  # [num_samples, batch]
    paths = perturbation_reshape(paths)  # [num_samples, batch, len1, len2, 9]
    pert_scores = tf.reduce_mean(scores, axis=0)
    pert_paths = tf.reduce_mean(paths, axis=0)
    if stop_paths_gradient:
      pert_paths = tf.stop_gradient(pert_paths)

    def grad(ds, dp):
      # Computes grad of scores w.r.t. (packed) sw_params.
      grad_scores = pert_paths * tf.reshape(ds, [tf.shape(ds)[0], 1, 1, 1])
      # Computes grad of scores w.r.t. sim_mat, gap_open and gap_extend.
      grad_scores = alignment.adjoint_weights_from_sim_mat(
          grad_scores, gap_open.shape, gap_extend.shape)
      if stop_paths_gradient:
        gradients = grad_scores
      else:
        # Flattens paths to shape [num_samples, batch, len1 * len2 * 9].
        flat_paths = tf.reshape(
            paths, [num_samples, tf.shape(paths)[1], -1])
        flat_noise_grad = tf.reshape(
            noise_grad, [num_samples, tf.shape(noise_grad)[1], -1])
        # Flattens dp to shape [num_samples, len1 * len2 * 9]
        flat_dp = tf.reshape(dp, [tf.shape(dp)[0], -1])
        # Computes grad of paths w.r.t. (packed) sw_params.
        grad_paths = tf.einsum('nbd,nb->bd', flat_noise_grad,
                               tf.einsum('nbd,bd->nb', flat_paths, flat_dp))
        grad_paths /= sigma * num_samples
        grad_paths = tf.reshape(grad_paths, shape)  # [batch, len1, len2, 9]
        # Computes grad of paths w.r.t. sim_mat, gap_open and gap_extend.
        grad_paths = alignment.adjoint_weights_from_sim_mat(
            grad_paths, gap_open.shape, gap_extend.shape)
        # Adds gradients w.r.t. scores and gradients w.r.t. paths.
        gradients = tf.nest.map_structure(
            lambda x, y: x + y, grad_scores, grad_paths)
      return gradients

    return (pert_scores, pert_paths), grad

  return forward((sim_mat, gap_open, gap_extend))
