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

"""FC DP sanitizer."""

import functools

import jax
import jax.numpy as jnp
import numpy as np


def clip(x, clip_norm=1.0):
  divisor = jnp.maximum(jnp.linalg.norm(x) / clip_norm, 1.)
  return x / divisor


def noisy(x, s, key):
  if 0 < s < np.inf:
    noise = jax.random.normal(key, shape=jnp.shape(x)) * s
    return x + noise
  return x


class Sanitizer(object):
  """Provides a set of functions for sanitizing private information.

  There are utilities used during pre-processing of the data, and others used
  during training. Sanitizer also provides a compute_epsilon function to compute
  the privacy loss for its sanitization process.

  Training:
    clip: clips the norm of each embedding. Used during row solves.
    apply_noise_gramian: adds noise to feature covariance.
    apply_noise: adds noise to the gradient.
  """

  def __init__(self,
               steps,
               max_norm=1,
               s0=0,
               random_seed=None):
    """Initializes a Sanitizer.

    Args:
      steps: number of gradient steps.
      max_norm: clips the user embeddings to max_norm.
      s0: the noise factor for fc vs grads.
      random_seed: seed.
    """
    if random_seed:
      self.key = jax.random.PRNGKey(random_seed)
    else:
      self.key = jax.random.PRNGKey(42)
    self.max_norm = max_norm
    self.sigmas = [s0, s0]
    self.steps = steps

  def refresh_key(self):
    """Use PRNG key only once, this function refreshes it once its used."""
    _, self.key = jax.random.split(self.key)

  def clip(self, embeddings):
    if not self.max_norm:
      return embeddings
    return jax.vmap(functools.partial(clip, clip_norm=self.max_norm))(
        embeddings)

  def _project_psd(self, x, rank):
    """Project to a PSD matrix."""
    if rank == 2:
      indices = [1, 0]
    elif rank == 3:
      indices = [0, 2, 1]
    else:
      raise ValueError("rank must be 2 or 3")

    def transpose(x):
      return jnp.transpose(x, indices)

    x = (x + transpose(x)) / 2
    e, v = jnp.linalg.eigh(x)
    e = jnp.maximum(e, 0)
    return v @ (jnp.expand_dims(e, -1) * transpose(v))

  def apply_noise_gramian(self, global_gramian):
    sigmas = self.sigmas
    max_norm = self.max_norm
    sigma = sigmas[0] * (max_norm**2)

    gram = noisy(global_gramian, sigma, key=self.key)
    gram = self._project_psd(gram, rank=2)
    return gram

  def apply_noise(self, rhs):
    if not self.max_norm:
      raise ValueError("max_norm must not be non-zero.")
    sigmas = self.sigmas
    max_norm = self.max_norm
    sigma = sigmas[1] * max_norm

    rhs_noised = noisy(rhs, sigma, key=self.key)
    return rhs_noised

  def compute_epsilon(self, target_delta):
    """Computes epsilon."""
    if not all(self.sigmas):
      return np.inf
    # The accounting is done as follows: whenever we compute a statistic with
    # L2 sensitivity k and add Gaussian noise of scale σ, the procedure is
    # (α, αβ/2)-RDP with β = k²/σ². To compose RDP processes, we sum their β.
    # computations:
    # - Gramian (k² = budget, σ = s1)
    # - RHSs (k² = budget, σ = s2)
    s1, s2 = self.sigmas
    beta = (1.0 / (s1**2) + self.steps / (s2**2))
    # We translate (α, αβ/2)-RDP to (ε, δ)-DP with ε = αβ/2 + log(1/δ)/(α−1).
    # We pick the α that minimizes ε, which is α = 1 + √(2log(1/δ)/β)
    alpha = 1.0 + np.sqrt(np.log(1.0 / target_delta) * 2.0 / beta)
    eps = alpha * beta / 2.0 + np.log(1.0 / target_delta) / (alpha - 1.0)
    return eps

  def set_sigmas(self,
                 target_epsilon,
                 target_delta,
                 sigma_ratio1=1):
    """Sets sigmas to get the target (epsilon, delta).

    Args:
      target_epsilon: the desired epsilon.
      target_delta: the desired delta.
      sigma_ratio1: the ratio sigma1/sigma2.
    """
    s_lower = 1e-6
    s_upper = 1e6

    def get_epsilon(s):
      self.sigmas = [sigma_ratio1 * s, s]
      return self.compute_epsilon(target_delta)

    eps = get_epsilon(s_lower)
    i = 0
    while np.abs(eps / target_epsilon - 1) > 0.0001:
      s = (s_lower + s_upper) / 2
      eps = get_epsilon(s)
      if eps > target_epsilon:
        s_lower = s
      else:
        s_upper = s
      i += 1
      if i > 1000:
        raise ValueError(
            f"No value of sigmas found for the desired (epsilon, delta)="
            f"={target_epsilon, target_delta}. Consider increasing stddev.")
    s1, s2 = self.sigmas
    print(
        f"Setting sigmas to [{s1:.2f}, {s2:.2f}], given target "
        f"(epsilon, delta)={target_epsilon, target_delta}")
