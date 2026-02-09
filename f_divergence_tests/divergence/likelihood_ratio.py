# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Implements f-divergence estimators based on the likelihood ratio."""

import jax.numpy as jnp
from f_divergence_tests import testing_typing
from f_divergence_tests.divergence import kernel as kernel_module


class LikelihoodRatioEstimator(kernel_module.KernelDivergence):
  """Estimates the likelihood ratio between two distributions using samples.

  Estimates the likelihood ratio function between two distributions using
  samples based on the estimator proposed in https://arxiv.org/abs/2409.14980
  for DrMMD. Specifically, it estimates the witness function for an
  interpolation between the chi-squared and MMD's witness functions. The witness
  function for the chi-squared is given by dP/dQ - 1.

  Attributes:
    lmbda: Regularization parameter. As lmbda goes to zero, it approaches the
      ratio. As lmbda goes to infinity, it approaches the witness function for
      the MMD.
  """

  def __init__(self, lmbda):
    super().__init__(has_vectorized_kernel=False)
    self.lmbda = lmbda

  def likelihood_ratio_yx_on_z(
      self,
      k_zx,
      k_zy,
      k_xx,
      inv_k_xx,
      k_xy,
  ):
    """Estimates the likelihood ratio between two distributions using samples.

    Uses DrMMD's estimator for the likelihood ratio using expression (71) in
    their paper.

    Args:
      k_zx: Kernel matrix of the first distribution.
      k_zy: Kernel matrix of the second distribution.
      k_xx: Kernel matrix of the first distribution.
      inv_k_xx: Inverse of the kernel matrix of the first distribution.
      k_xy: Kernel matrix of the interpolation.

    Returns:
      The likelihood ratio between the two distributions.
    """
    first_term = k_zy.mean(axis=1) - k_zx.mean(axis=1)
    second_term = -(k_zx @ inv_k_xx @ k_xy).mean(axis=1)
    third_term = k_zx @ inv_k_xx @ k_xx.mean(axis=1)

    chi_square_vals = (
        (first_term + second_term + third_term).squeeze()
        / self.lmbda
        * 2
        * (1 + self.lmbda)
    )
    likelihood_vals = chi_square_vals + 1
    return likelihood_vals


class DrMMDSymmetric(kernel_module.KernelDivergence):
  """Computes the De-regularized maximum mean discrepancy.

  De-regularized maximum mean discrepancy (DrMMD) is introduced in
  https://arxiv.org/pdf/2409.14980. It combines the chi-squared and the MMD.

  Attributes:
    lmbda: Regularization parameter. As lmbda goes to zero, DrMMD approaches the
      chi-squared divergence. As lmbda goes to infinity, DrMMD approaches the
      MMD.
  """

  def __init__(self, lmbda):
    super().__init__(has_vectorized_kernel=False)
    self.lmbda = lmbda

  def _drmmd(
      self, k_xx, k_yy, k_xy
  ):
    _, m = k_yy.shape[0], k_xx.shape[0]
    inv_k_xx = jnp.linalg.inv(k_xx + m * self.lmbda * jnp.eye(k_xx.shape[0]))

    part1 = k_yy.mean() + k_xx.mean() - 2 * k_xy.mean()
    part2 = -(k_xy.T @ inv_k_xx @ k_xy).mean()
    part3 = (k_xx.T @ inv_k_xx @ k_xy).mean() * 2
    part4 = -(k_xx.T @ inv_k_xx @ k_xx).mean()

    return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)

  def __call__(
      self, k_xx, k_yy, k_xy
  ):
    p_vs_q = self._drmmd(k_xx, k_yy, k_xy)
    q_vs_p = self._drmmd(k_yy, k_xx, k_xy.T)

    return jnp.maximum(p_vs_q, q_vs_p)


class KernelFDivergence(LikelihoodRatioEstimator):
  """Base class for kernel f-divergence estimators.

  This class is a base class for kernel f-divergence estimators. It is a wrapper
  around the likelihood ratio estimator that computes the f-divergence from the
  likelihood ratio using the following identity:
    D_f(P||Q) = E_x[f'(dP/dQ(x))] - E_y[f_star(f'(dP/dQ(y)))]
  where f_star is the convex conjugate of f and f' is the first derivative of
  the f function.

  Attributes:
    ratio_postprocess_fn: first derivative of the f function defining the
      f-divergence.Function that maps the likelihood ratio to the divergence.
    conjugate_fn: Function that maps the divergence to the convex conjugate.
  """

  def __init__(self, lmbda, ratio_postprocess_fn, conjugate_fn):
    super().__init__(lmbda)
    self.ratio_postprocess_fn = ratio_postprocess_fn
    self.conjugate_fn = conjugate_fn

  def __call__(
      self, k_xx, k_yy, k_xy
  ):
    """Estimates f-divergence from variational form and likelihood ratio.

    Specifically, it estimates the f-divergence from formula (1) in paper:
                sum_x g(x) - sum_y f_star(g(x))
    with g=ratio_postprocess_fn and f_star=conjugate_fn.

    Args:
      k_xx: Kernel matrix of the first distribution.
      k_yy: Kernel matrix of the second distribution.
      k_xy: Kernel matrix of the cross distributions.

    Returns:
      The f-divergence between the two distributions.
    """
    _, m = k_yy.shape[0], k_xx.shape[0]

    inv_k_xx = jnp.linalg.inv(k_xx + m * self.lmbda * jnp.eye(m))

    dq_dp_x = self.likelihood_ratio_yx_on_z(
        k_zx=k_xx, k_zy=k_xy, k_xx=k_xx, inv_k_xx=inv_k_xx, k_xy=k_xy
    )
    dq_dp_y = self.likelihood_ratio_yx_on_z(
        k_zx=k_xy.T, k_zy=k_yy, k_xx=k_xx, inv_k_xx=inv_k_xx, k_xy=k_xy
    )
    divergence_q_p = jnp.mean(self.ratio_postprocess_fn(dq_dp_y)) - jnp.mean(
        self.conjugate_fn(self.ratio_postprocess_fn(dq_dp_x))
    )

    return divergence_q_p


class HockeyStickKernelFDivergence(KernelFDivergence):
  """Hockey-stick divergence from variational form and likelihood ratio."""

  def __init__(self, lmbda, order = 1.0):
    g = lambda x: x > order
    f_star = lambda u: jnp.where(
        (u <= 1) & (u >= 0),  # Condition: evaluated for each element
        u * order,  # Value if condition is True
        jnp.inf,  # Value if condition is False
    )
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class KullbackLeiblerFDivergence(KernelFDivergence):
  """Kullback-Leibler (KL) divergence from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = 1 + log(r)
    g = lambda r: 1.0 + jnp.log(r)
    # Corresponds to f*(u) = exp(u - 1)
    f_star = lambda u: jnp.exp(u - 1.0)
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class ReverseKLFDivergence(KernelFDivergence):
  """Reverse KL divergence from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = 1 - 1/r
    g = lambda r: 1.0 - (1.0 / r)
    # Corresponds to f*(u) = -log(1 - u) for u < 1
    f_star = lambda u: jnp.where(u < 1.0, -jnp.log(1.0 - u), jnp.inf)
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class JeffreysFDivergence(KernelFDivergence):
  """Jeffreys (symmetrized KL) divergence from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = log(r) + (r-1)/r
    g = lambda r: jnp.log(r) + (r - 1.0) / r

    # f*(u) does not have a simple closed-form expression from the table.
    def f_star(u):
      raise NotImplementedError(
          "The convex conjugate for Jeffreys divergence is not available "
          "in a simple closed form."
      )

    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class JensenShannonFDivergence(KernelFDivergence):
  """Jensen-Shannon divergence from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = log(2r / (r+1))
    g = lambda r: jnp.log(2.0 * r / (r + 1.0))
    # Corresponds to f*(u) = log(1 + exp(u)) - log(2)
    f_star = lambda u: jnp.log(1.0 + jnp.exp(u)) - jnp.log(2.0)
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class TotalVariationFDivergence(KernelFDivergence):
  """Total Variation distance from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = sign(r-1)/2
    g = lambda r: jnp.sign(r - 1.0) / 2.0
    # Corresponds to f*(u) = u for |u| <= 1/2
    f_star = lambda u: jnp.where(jnp.abs(u) <= 0.5, u, jnp.inf)
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class PearsonChiSquaredFDivergence(KernelFDivergence):
  """Pearson Chi-squared divergence from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = 2(r-1)
    g = lambda r: 2.0 * (r - 1.0)
    # Corresponds to f*(u) = u + u^2 / 4
    f_star = lambda u: u + (u**2) / 4.0
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


class SquaredHellingerFDivergence(KernelFDivergence):
  """Squared Hellinger distance from variational form."""

  def __init__(self, lmbda):
    # Corresponds to g*(x) = (sqrt(r)-1)/sqrt(r) = 1 - 1/sqrt(r)
    g = lambda r: 1.0 - 1.0 / jnp.sqrt(r)
    # Corresponds to f*(u) = u / (1-u) for u < 1
    f_star = lambda u: jnp.where(u < 1.0, u / (1.0 - u), jnp.inf)
    super().__init__(lmbda, ratio_postprocess_fn=g, conjugate_fn=f_star)


def symmetric_estimator_factory(class_):

  class SymmetricEstimator(class_):

    def __call__(self, k_xx, k_yy, k_xy):
      dq_dp = super().__call__(k_xx, k_yy, k_xy)  # pytype: disable=attribute-error
      dp_dq = super().__call__(k_yy, k_xx, k_xy.T)  # pytype: disable=attribute-error
      return jnp.maximum(dp_dq, dq_dp)

  return SymmetricEstimator


SymmetricHockeyStick = symmetric_estimator_factory(HockeyStickKernelFDivergence)
SymmetricKLDivergence = symmetric_estimator_factory(KullbackLeiblerFDivergence)
SymmetricReverseKL = symmetric_estimator_factory(ReverseKLFDivergence)
SymmetricJensenShannon = symmetric_estimator_factory(JensenShannonFDivergence)
SymmetricTotalVariation = symmetric_estimator_factory(TotalVariationFDivergence)
SymmetricPearsonChiSquared = symmetric_estimator_factory(
    PearsonChiSquaredFDivergence
)
SymmetricSquaredHellinger = symmetric_estimator_factory(
    SquaredHellingerFDivergence
)

