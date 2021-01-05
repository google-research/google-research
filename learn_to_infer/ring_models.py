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

"""Transformer models for performing inference in a mixture of rings model.
"""
from functools import partial

from . import ring_dist
from . import transformer
from . import util

import flax
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random


class RingInferenceMachine(object):
  """Model which predicts cluster means from a batch of data."""

  def __init__(self,
               max_k=2,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               weight_init=jax.nn.initializers.xavier_uniform()):
    """Creates the model.

    Args:
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      num_heads: The number of heads to use in the transformer.
      num_encoders: The number of encoder layers to use in the transformer.
      num_decoders: The number of decoder layers to use in the transformer.
      qkv_dim: The dimensions of the queries, keys, and values in the
        transformer.
      activation_fn: The activation function to use for hidden layers.
      weight_init: The weight initializer.
    """
    self.max_k = max_k
    self.max_num_data_points = max_num_data_points
    self.tfmr = transformer.EncoderDecoderTransformer.partial(
        target_dim=5,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, weight_init=weight_init)

  def init_params(self, key):
    """Initializes the parameters of the model using dummy data.

    Args:
      key: A JAX PRNG key
    Returns:
      params: The parameters of the model.
    """
    batch_size = 1
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(
        subkey, [batch_size, self.max_num_data_points, 2])
    input_lengths = jnp.full([batch_size], self.max_num_data_points)
    ks = jnp.full([batch_size], self.max_k)
    _, params = self.tfmr.init(key, inputs, input_lengths, ks)
    return params

  def loss(self, params, inputs, input_lengths, true_params, ks, key):
    """Computes the wasserstein loss for this model.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers representing the number of
        data points in each batch element.
      true_params: A three-tuple containing
        r_means: A [batch_size, max_k] tensor containing the true means of
          the radii.
        r_scales: A [batch_size, max_k] tensor containing the true scales of the
          radii.
        centers: A [batch_size, max_k, 2] tensor containing the centers of each
          ring.
        log_weights: A [batch_size, max_k] tensor containing the true log
          weights of each ring.
      ks: A [batch_size] set of integers representing the true number of
        clusters in each batch element.
      key: A JAX PRNG key.
    Returns:
      The wasserstein distance from the set of predicted mus to the true set
      of mus, a tensor of shape [batch_size].
    """
    rad_means, rad_scales, centers, log_weights = true_params
    targets = jnp.concatenate([jnp.log(rad_means[Ellipsis, jnp.newaxis]),
                               jnp.log(rad_scales[Ellipsis, jnp.newaxis]),
                               centers,
                               log_weights[Ellipsis, jnp.newaxis]], axis=2)
    return self.tfmr.wasserstein_distance_loss(
        params, inputs, input_lengths, targets, ks, key)

  def predict(self, params, inputs, input_lengths, ks):
    """Predicts the cluster means for the given data sets.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      The predicted means, a tensor of shape [batch_size, max_k, data_dim].
    """
    raw_outs = self.tfmr.call(params, inputs, input_lengths, ks)
    rad_means = jnp.exp(raw_outs[:, :, 0])
    rad_scales = jnp.exp(raw_outs[:, :, 1])
    centers = raw_outs[:, :, 2:4]
    log_weights = raw_outs[:, :, 4]

    return (rad_means, rad_scales, centers, log_weights)

  def classify(self, params, inputs, input_lengths, ks):
    """Assigns each point to cluster based on the predicted cluster means.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, 2] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      The predicted clusters, an integer tensor of shape
        [batch_size, max_num_data_points]. Each element is in [0, max_num_k).
    """
    rad_means, rad_scales, centers, log_weights = self.predict(
        params, inputs, input_lengths, ks)

    # Compute each point's log prob under each ring distribution
    log_ps = vmap(vmap(
        vmap(ring_dist.ring_log_p, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0)))(inputs, rad_means, rad_scales, centers)
    log_ps = log_ps + log_weights[Ellipsis, jnp.newaxis]
    log_ps = jnp.where(
        util.make_mask(ks, self.max_k)[:, :, jnp.newaxis], log_ps,
        jnp.full_like(log_ps, -jnp.inf))
    clusters = jnp.argmax(log_ps, axis=-2)
    return clusters, (rad_means, rad_scales, centers, log_weights)
