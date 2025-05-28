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

"""Random variable utilities."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from d3pm.insertdelete import math_util

NDArray = Any


def random_geometric(key, continue_log_prob):
  """Sample a geometric r.v. with given log prob of continuing."""
  return jnp.floor(jnp.log(jax.random.uniform(key)) / continue_log_prob).astype(
      jnp.int32)


def log_choose(n, k):
  """Computes log(n choose k)."""
  return jnp.where(
      (k >= 0) & (n >= k),
      jax.scipy.special.gammaln(n + 1) - jax.scipy.special.gammaln(k + 1) -
      jax.scipy.special.gammaln(n - k + 1), -jnp.inf)


@flax.struct.dataclass
class RandomVariablePDF:
  """Wrapper for a PDF of a random variable."""
  log_probs: NDArray

  @staticmethod
  def from_log_pdf(n, fn):
    return RandomVariablePDF(jax.vmap(fn)(jnp.arange(n)))

  def shift(self, by):
    return RandomVariablePDF(
        jnp.where(
            jnp.arange(self.log_probs.shape[0]) < by, -jnp.inf,
            jnp.roll(self.log_probs, by)))

  def mixture_of(self, batched_rv):
    safe_batched_log_probs = jnp.where(self.log_probs[:, None] == -np.inf,
                                       -np.inf, batched_rv.log_probs)
    return RandomVariablePDF(
        math_util.safe_logsumexp(
            self.log_probs[:, None] + safe_batched_log_probs, axis=0))

  @property
  def probs(self):
    return jnp.exp(self.log_probs)


def _logpowexp(lp, k):
  """log(exp(lp)^k)."""
  return jnp.where(k == 0, 0.0, k * lp)


def binomial_log_pdf(num_elements,
                     success_log_prob,
                     failure_log_prob=None,
                     max_len=None):
  """PDF of a binomial RV."""
  if max_len is None:
    max_len = num_elements
  # https://en.wikipedia.org/wiki/Negative_binomial_distribution
  # where p = exp(continue_log_prob), r = num_failures
  if failure_log_prob is None:
    log_p = success_log_prob
    log_1m_p = jnp.log1p(-jnp.exp(success_log_prob))
  else:
    total_log_prob = math_util.safe_logaddexp(success_log_prob,
                                              failure_log_prob)
    log_p = math_util.safe_sub_or_ninf(success_log_prob, total_log_prob)
    log_1m_p = math_util.safe_sub_or_ninf(failure_log_prob, total_log_prob)

  n = num_elements

  def log_pdf(k):
    return jnp.where(
        (n >= k) & (k >= 0),
        (log_choose(n, k) + _logpowexp(log_1m_p, n - k) + _logpowexp(log_p, k)),
        -jnp.inf)

  return RandomVariablePDF.from_log_pdf(max_len, log_pdf)


def geometric_log_pdf(continue_log_prob, max_len):
  """(Truncated) PDF of a geometric RV. By convention includes 0."""
  log_p = continue_log_prob
  log_1m_p = jnp.log1p(-jnp.exp(continue_log_prob))
  return RandomVariablePDF.from_log_pdf(
      max_len, lambda k: (_logpowexp(log_p, k) + log_1m_p))


def negative_binomial_log_pdf(continue_log_prob, num_failures, max_len):
  """(Truncated) PDF of a negative binomial RV. By convention includes 0."""
  # https://en.wikipedia.org/wiki/Negative_binomial_distribution
  # where p = exp(continue_log_prob), r = num_failures
  r = num_failures
  log_p = continue_log_prob
  log_1m_p = jnp.log1p(-jnp.exp(continue_log_prob))

  def _log_pdf_fn(k):
    return jnp.where(
        r == 0,
        jnp.log(k == 0),
        (
            log_choose(k + r - 1, k)
            + _logpowexp(log_1m_p, r)
            + _logpowexp(log_p, k)
        ),
    )

  return RandomVariablePDF.from_log_pdf(max_len, _log_pdf_fn)


def negative_hypergeometric_log_pdf(success_elts,
                                    failure_elts,
                                    num_failures,
                                    max_len=None):
  """PDF of a negative hypergeometric RV.

  Measures the number of successes before the first `num_failures` failures
  when drawing from a fixed number of possible successes and failures.

  Args:
    success_elts: How many chances there are to succeed.
    failure_elts: How many chances there are to fail.
    num_failures: How many failures we stop at.
    max_len: How many indices of the PDF to compute.

  Returns:
    Array of PDF values.
  """
  # https://en.wikipedia.org/wiki/Negative_binomial_distribution
  # https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution
  N = success_elts + failure_elts  # pylint:disable=invalid-name
  K = success_elts  # pylint:disable=invalid-name
  r = num_failures
  if max_len is None:
    max_len = N

  def log_pdf(k):
    return jnp.where(
        r == 0, jnp.log(k == 0),
        log_choose(k + r - 1, k) + log_choose(N - r - k, K - k) -
        log_choose(N, K))

  return RandomVariablePDF.from_log_pdf(max_len, log_pdf)
