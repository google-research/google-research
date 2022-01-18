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

"""Implementation of Gadget 2."""

from typing import List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from gumbel_max_causal_gadgets import coupling_util


def fix_coupling_sinkhorn(log_coupling, log_p1, log_p2, iterations=10):
  """Adjust a coupling to approximately match marginals using Sinkhorn iteration.

  Args:
    log_coupling: Log of a coupling, possibly unnormalized, of shape [M,M]
    log_p1: Log probabilities for first dimension logits, of shape [M].
    log_p2: Log probabilities for second dimension logits, of shape [M].
    iterations: How many Sinkhorn iterations to use.

  Returns:
    Matrix of shape [M,M]. Will always exactly match the marginals of `log_p2`
    along the second axis, and will attempt to match marginals of `log_p1` along
    the first axis as well, but may not reach it depending on iteration count.
  """
  for _ in range(iterations):
    log_coupling = (
        log_coupling + log_p1[:, None] -
        jax.scipy.special.logsumexp(log_coupling, axis=1, keepdims=True))
    log_coupling = (
        log_coupling + log_p2[None, :] -
        jax.scipy.special.logsumexp(log_coupling, axis=0, keepdims=True))
  return log_coupling


def fix_coupling_rejection(log_coupling, log_p1, log_p2):
  """Apply final correction step to ensure a coupling satisfies both marginals.

  Args:
    log_coupling: Log of a coupling, possibly unnormalized, of shape [M,M].
      Assumed to approximately be a coupling of log_p1 and log_p2.
    log_p1: Log probabilities for first dimension logits, of shape [M].
    log_p2: Log probabilities for second dimension logits, of shape [M].

  Returns:
    A matrix of shape [M,M] representing a valid coupling of log_p1 and log_p2.
    If the original coupling was already close to a valid coupling, it will
    be only changed slightly.
  """
  # Normalize so that it matches p1. Then consider mixing with a p-independent
  # distribution so that it also matches p2.
  log_coupling_fixed_p1 = (
      log_coupling + log_p1[:, None] -
      jax.scipy.special.logsumexp(log_coupling, axis=1, keepdims=True))
  approx_log_p2 = jax.scipy.special.logsumexp(log_coupling_fixed_p1, axis=0)
  # How much more often do we sample each value of s2 than we should?
  # accept rate = C p2(x)/p2tilde(x)
  accept_rate = log_p2 - approx_log_p2
  accept_rate = accept_rate - jnp.max(accept_rate)
  # Downweight rejections.
  log_coupling_accept_s2_given_s1 = jax.nn.log_softmax(
      log_coupling_fixed_p1, axis=-1) + accept_rate[None, :]
  # Compensate by then drawing from p2 exactly if we failed.
  log_prob_keep = jax.scipy.special.logsumexp(
      log_coupling_accept_s2_given_s1, axis=-1)
  # print(accept_rate, log_prob_keep)
  certainly_keep = jnp.exp(log_prob_keep) >= 1.0
  resample_log_p1rob = jnp.where(
      certainly_keep, -jnp.inf,
      jnp.log1p(-jnp.where(certainly_keep, 0.0, jnp.exp(log_prob_keep))))
  compensation = resample_log_p1rob[:, None] + log_p2[None, :]
  return log_p1[:, None] + jnp.logaddexp(log_coupling_accept_s2_given_s1,
                                         compensation)


class GadgetTwoMLPPredictor(nn.Module):
  """Gadget 2 coupling layer, supporting counterfactual inference.

  Attributes:
    S_dim: Number of possible outcomes.
    Z_dim: Size of the latent space. May need to be much larger than S_dim to
      obtain good results.
    hidden_features: List of dimensions for hidden layers.
    sinkhorn_iterations: Number of Sinkhorn iterations to use before final
      correction step.
    learn_prior: Whether to create parameters for the prior. Note that we only
      compute derivatives with respect to these when using
      `counterfactual_sample_relaxed`, which is not supported by gadget 1. We
      set this to False for all of our experiments.
    relaxation_temperature: Default temperature used when training.
  """
  S_dim: int  # pylint: disable=invalid-name
  Z_dim: int  # pylint: disable=invalid-name
  hidden_features: List[int]
  sinkhorn_iterations: int = 10
  learn_prior: bool = True
  relaxation_temperature: Optional[float] = None

  def setup(self):
    self.hidden_layers = [nn.Dense(feats) for feats in self.hidden_features]
    self.output_layer = nn.DenseGeneral((self.Z_dim, self.S_dim))
    if self.learn_prior:
      self.prior = self.param("prior", nn.zeros, (self.Z_dim,))

  def get_prior(self):
    """Returns the learned or static prior distribution."""
    if self.learn_prior:
      return jax.nn.log_softmax(self.prior)
    else:
      return jax.nn.log_softmax(jnp.zeros([self.Z_dim]))

  def get_joint(self, s_logits):
    """Returns a joint that agrees with s_logits when axis 0 is marginalized out."""
    z_logits = self.get_prior()
    value = jax.nn.softmax(s_logits)
    for layer in self.hidden_layers:
      value = nn.sigmoid(layer(value))

    log_joint = self.output_layer(value)
    # Sinkhorn has nicer gradients.
    log_joint = fix_coupling_sinkhorn(log_joint, z_logits, s_logits,
                                      self.sinkhorn_iterations)
    # ... but rejection is harder to exploit inaccuracy for
    log_joint = fix_coupling_rejection(log_joint, z_logits, s_logits)
    return log_joint

  def get_forward(self, s_logits):
    """Returns a matrix corresponding to `log pi(x | z, s_logits)`."""
    return jax.nn.log_softmax(self.get_joint(s_logits), axis=-1)

  __call__ = get_joint

  def sample(self, s_logits, rng):
    """Draws a single sample from `s_logits`.

    Args:
      s_logits: Logits to sample from.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling.

    Returns:
      Sampled integer index from s_logits.
    """
    log_z = self.get_prior()
    log_s_given_z = self.get_forward(s_logits)
    k1, k2 = jax.random.split(rng)
    z = jax.random.categorical(k1, log_z)
    # jax categorical does Gumbel-max internally, so given common random numbers
    # this will produce a Gumbel-max coupling of s given z.
    s = jax.random.categorical(k2, log_s_given_z[z])
    return s

  def sample_relaxed(self, s_logits, rng, temperature=None):
    """Sample a relaxed Gumbel-softmax output.

    Args:
      s_logits: Logits to sample from.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling.
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
    log_z = self.get_prior()
    log_s_given_z = self.get_forward(s_logits)
    k1, k2 = jax.random.split(rng)
    z = jax.random.categorical(k1, log_z)
    g = jax.random.gumbel(k2, [self.S_dim])
    # Gumbel-softmax instead of argmax here
    return jax.nn.softmax((g + log_s_given_z[z]) / temperature)

  def counterfactual_sample(self, p_logits, q_logits, p_observed, rng):
    """Sample a single sample from q conditioned on observing p_observed.

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
    log_z = self.get_prior()
    log_ps_given_z = self.get_forward(p_logits)
    log_qs_given_z = self.get_forward(q_logits)
    # Infer z from p_observed
    log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, p_observed]
    z = jax.random.categorical(k1, log_z_given_ps)
    # Infer Gumbels from p_observed and z
    gumbels = coupling_util.counterfactual_gumbels(log_ps_given_z[z],
                                                   p_observed, k2)
    # Choose accordingly
    qs = jnp.argmax(gumbels + log_qs_given_z[z])
    return qs

  def counterfactual_sample_relaxed(self,
                                    p_logits,
                                    q_logits,
                                    p_observed,
                                    rng,
                                    temperature=1.0):
    """Sample a relaxed sample from q conditioned on observing p_observed.

    This essentially continuously relaxes the counterfactual outcome without
    relaxing the observed outcome. This should enable learning the prior over z
    in addition to the transformation. We did not use this in our experiments.

    Args:
      p_logits: Logits describing the original distribution of p_observed.
      q_logits: Logits describing a new counterfactual intervention.
      p_observed: Sample index we observed.
      rng: PRNGKey. Sharing this across multiple calls produces an implicit
        coupling.
      temperature: Relaxation temperature to use. Defaults to the value from the
        class.

    Returns:
      Vector float32[S_dim] of relaxed outputs that sum to 1.
    """
    log_z = self.get_prior()
    log_ps_given_z = self.get_forward(p_logits)
    log_qs_given_z = self.get_forward(q_logits)
    # Infer distn of z from p_observed.
    log_z_given_ps = (log_z[:, None] + log_ps_given_z)[:, p_observed]
    z_given_ps = jax.nn.softmax(log_z_given_ps)

    # For each z, counterfactually sample some Gumbels, then apply softmax.
    def relaxed_for_fixed_z(z):
      # (again, uses common random numbers here)
      g = coupling_util.counterfactual_gumbels(log_ps_given_z[z], p_observed,
                                               rng)
      # Gumbel-softmax instead of argmax here
      soft_q = jax.nn.softmax((g + log_qs_given_z[z]) / temperature)
      return soft_q

    distns = jax.vmap(relaxed_for_fixed_z)(jnp.arange(self.Z_dim))
    avg_distn = jnp.sum(z_given_ps[:, None] * distns, axis=0)
    return avg_distn
