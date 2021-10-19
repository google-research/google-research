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

"""Implementation of Gadget 1."""

from typing import List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from gumbel_max_causal_gadgets import coupling_util


class GadgetOneMLPPredictor(nn.Module):
  """Gadget 1 coupling layer, supporting counterfactual inference.

  Attributes:
    S_dim: Number of possible outcomes.
    hidden_features: List of dimensions for hidden layers.
    relaxation_temperature: Default temperature used when training.
  """
  S_dim: int  # pylint: disable=invalid-name
  hidden_features: List[int]
  relaxation_temperature: Optional[float] = None

  def setup(self):
    self.hidden_layers = [nn.Dense(feats) for feats in self.hidden_features]
    self.output_layer = nn.DenseGeneral((self.S_dim, self.S_dim))

  def get_joint(self, s_logits):
    """Returns a joint that agrees with s_logits when axis 0 is marginalized out."""
    value = jax.nn.softmax(s_logits)
    for layer in self.hidden_layers:
      value = nn.sigmoid(layer(value))

    log_joint = self.output_layer(value)

    # Fix the marginals for the output.
    log_joint = (
        log_joint -
        jax.scipy.special.logsumexp(log_joint, axis=0, keepdims=True) +
        s_logits[None, :])

    return log_joint

  __call__ = get_joint

  def sample(self, s_logits, rng, transpose=False):
    """Draws a single sample from `s_logits`.

    Args:
      s_logits: Logits to sample from.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling, but `transpose` should be passed to one of them.
      transpose: Whether or not to transpose the exogenous noise.

    Returns:
      Sampled integer index from s_logits.
    """
    log_joint = self.get_joint(s_logits)
    gumbels = jax.random.gumbel(rng, log_joint.shape)
    if transpose:
      gumbels = gumbels.T
    shifted_gumbels = log_joint + gumbels
    max_shifted_gumbels_over_alternate = jnp.max(shifted_gumbels, axis=0)
    s_sample = jnp.argmax(max_shifted_gumbels_over_alternate)
    return s_sample

  def sample_relaxed(self, s_logits, rng, transpose=False, temperature=None):
    """Sample a relaxed Gumbel-softmax output.

    Args:
      s_logits: Logits to sample from.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling.
      transpose: Whether or not to transpose the exogenous noise.
      temperature: Relaxation temperature to use. Defaults to the value from the
        class.

    Returns:
      Vector float32[S_dim] of relaxed outputs that sum to 1. The argmax of
      this vector will always be the same as the result of the equivalent call
      to `sample`.
    """
    if temperature is None:
      assert self.relaxation_temperature
      temperature = self.relaxation_temperature
    log_joint = self.get_joint(s_logits)
    gumbels = jax.random.gumbel(rng, log_joint.shape)
    if transpose:
      gumbels = gumbels.T
    shifted_gumbels = log_joint + gumbels
    # First take a hard max
    max_shifted_gumbels_over_alternate = jnp.max(shifted_gumbels, axis=0)
    # Then take a soft max
    return jax.nn.softmax(max_shifted_gumbels_over_alternate / temperature)

  def counterfactual_sample(self, p_logits, q_logits, p_observed, rng):
    """Sample a single sample from q conditioned on observing p_observed.

    Automatically transposes the noise as needed in order to compute the
    counterfactual sample.

    Args:
      p_logits: Logits describing the original distribution of p_observed.
      q_logits: Logits describing a new counterfactual intervention.
      p_observed: Sample index we observed.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling.

    Returns:
      Sampled integer index from q_logits, conditioned on observing p_observed
      under p_logits.
    """
    k1, k2 = jax.random.split(rng)
    log_joint_from_p = self.get_joint(p_logits)
    log_joint_from_q = self.get_joint(q_logits).T  # transpose!

    # Sample what p "thought" q was.
    q_hat_from_p = jax.random.categorical(k1, log_joint_from_p[:, p_observed])

    # Sample the argmax under q's estimate, given that this was the argmax under
    # p's estimate
    flat_observed = q_hat_from_p * self.S_dim + p_observed
    log_joint_from_p_flat = jnp.reshape(log_joint_from_p, [-1])
    # log_joint_from_q_flat = jnp.reshape(log_joint_from_q, [-1])
    gumbels_flat = coupling_util.counterfactual_gumbels(log_joint_from_p_flat,
                                                        flat_observed, k2)
    gumbels = gumbels_flat.reshape([self.S_dim, self.S_dim])

    # Take the argmax for q.
    shifted_gumbels_for_q = gumbels + log_joint_from_q
    max_shifted_gumbels_over_p = jnp.max(shifted_gumbels_for_q, axis=1)
    q_sample = jnp.argmax(max_shifted_gumbels_over_p)
    return q_sample
