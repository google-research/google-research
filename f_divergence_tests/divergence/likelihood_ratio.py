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

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from f_divergence_tests import testing_typing
from f_divergence_tests.divergence import kernel as kernel_module


def regularized_cho_factor(
    k_xx, lmbda
):
  """Computes the Cholesky factorization of `k_xx + m * lmbda * I`.

  Args:
    k_xx: Kernel matrix of the denominator distribution x, shape `[m, m]`.
    lmbda: Positive regularization parameter.

  Returns:
    A `(factor, lower)` tuple as returned by `jax.scipy.linalg.cho_factor`,
    where `factor` has shape `[m, m]`.

  Example:
    ```python
    k_xx = jnp.eye(4, dtype=jnp.float32)
    cho_x = regularized_cho_factor(k_xx, lmbda=0.1)
    ```
  """
  m = k_xx.shape[0]
  return jsl.cho_factor(k_xx + m * lmbda * jnp.eye(m, dtype=k_xx.dtype))


class LikelihoodRatioEstimator(kernel_module.KernelDivergence):
  """Estimates the likelihood ratio dP_y/dP_x between two distributions.

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
    """Initializes the instance.

    Args:
      lmbda: Positive regularization parameter.
    """
    super().__init__(has_vectorized_kernel=False)
    self.lmbda = lmbda

  def likelihood_ratio_yx_on_z(
      self,
      k_zx,
      k_zy,
      k_xx,
      cho_factor_k_xx,
      k_xy,
  ):
    """Estimates the likelihood ratio dP_y/dP_x at query points z.

    Uses DrMMD's estimator for the likelihood ratio (expression (71) in
    https://arxiv.org/abs/2409.14980), where the x-distribution is the
    denominator and the result estimates dP_y/dP_x at z.

    Args:
      k_zx: Kernel matrix between queries z and denominator x, shape `[n_z, m]`.
      k_zy: Kernel matrix between queries z and numerator y, shape `[n_z, n]`.
      k_xx: Kernel matrix of the denominator distribution x, shape `[m, m]`.
      cho_factor_k_xx: Cholesky factor from `regularized_cho_factor(k_xx,
        self.lmbda)`.
      k_xy: Cross-kernel matrix between x and y, shape `[m, n]`.

    Returns:
      Estimated likelihood ratio dP_y/dP_x at z, shape `[n_z]`.
    """
    first_term = k_zy.mean(axis=1) - k_zx.mean(axis=1)
    second_term = -(k_zx @ jsl.cho_solve(cho_factor_k_xx, k_xy.mean(axis=1)))
    third_term = k_zx @ jsl.cho_solve(cho_factor_k_xx, k_xx.mean(axis=1))
    chi_square_vals = (
        (first_term + second_term + third_term)
        / self.lmbda
        * 2.0
        * (1.0 + self.lmbda)
    )
    return chi_square_vals + 1.0


class DrMMDSymmetric(LikelihoodRatioEstimator):
  """Computes the De-regularized maximum mean discrepancy.

  De-regularized maximum mean discrepancy (DrMMD) is introduced in
  https://arxiv.org/pdf/2409.14980. It combines the chi-squared and the MMD.

  Attributes:
    lmbda: Regularization parameter. As lmbda goes to zero, DrMMD approaches the
      chi-squared divergence. As lmbda goes to infinity, DrMMD approaches the
      MMD.
    symmetric: Whether `__call__` takes the maximum over both directions.
  """

  symmetric: bool = True

  def directional(
      self, k_xx, k_yy, k_xy
  ):
    """Estimates directional DrMMD D(P_y || P_x) with x as the denominator.

    Args:
      k_xx: Kernel matrix of the denominator distribution x, shape `[m, m]`.
      k_yy: Kernel matrix of the numerator distribution y, shape `[n, n]`.
      k_xy: Cross-kernel matrix between x and y, shape `[m, n]`.

    Returns:
      Scalar directional DrMMD estimate.
    """
    cho_x = regularized_cho_factor(k_xx, self.lmbda)
    k_xx_mean = k_xx.mean(axis=1)
    k_xy_mean = k_xy.mean(axis=1)
    sol_xy = jsl.cho_solve(cho_x, k_xy_mean)
    sol_xx = jsl.cho_solve(cho_x, k_xx_mean)

    part1 = k_yy.mean() + k_xx.mean() - 2.0 * k_xy.mean()
    part2 = -(k_xy_mean @ sol_xy)
    part3 = (k_xx_mean @ sol_xy) * 2.0
    part4 = -(k_xx_mean @ sol_xx)

    return (part1 + part2 + part3 + part4) / self.lmbda * (1.0 + self.lmbda)

  def __call__(
      self, k_xx, k_yy, k_xy
  ):
    """Estimates DrMMD between two distributions.

    Args:
      k_xx: Kernel matrix of the first distribution, shape `[m, m]`.
      k_yy: Kernel matrix of the second distribution, shape `[n, n]`.
      k_xy: Cross-kernel matrix between the two distributions, shape `[m, n]`.

    Returns:
      Scalar DrMMD estimate.
    """
    d = self.directional(k_xx, k_yy, k_xy)
    if not self.symmetric:
      return d
    return jnp.maximum(d, self.directional(k_yy, k_xx, k_xy.T))


class KernelFDivergence(LikelihoodRatioEstimator):
  """Base class for kernel f-divergence estimators.

  This class is a base class for kernel f-divergence estimators. It is a wrapper
  around the likelihood ratio estimator that computes the f-divergence from the
  likelihood ratio using the variational identity:
    D_f(P_y || P_x) = E_y[g(r(y))] - E_x[f_star(g(r(x)))]
  where r = dP_y/dP_x, f_star is the convex conjugate of f, and g = f' is the
  first derivative of the f function.

  Attributes:
    lmbda: Regularization parameter.
    ratio_postprocess_fn: First derivative of the f function defining the
      f-divergence. Function that maps the likelihood ratio to the divergence.
    conjugate_fn: Function that maps the divergence to the convex conjugate.
    symmetric: Class attribute indicating whether `__call__` takes the maximum
      over both directions; overridden by symmetric variants.
  """

  symmetric: bool = False

  def __init__(self, lmbda, ratio_postprocess_fn, conjugate_fn):
    """Initializes the instance.

    Args:
      lmbda: Positive regularization parameter.
      ratio_postprocess_fn: First derivative `g = f'` of the f-divergence
        generator.
      conjugate_fn: Convex conjugate `f*` of the f-divergence generator.
    """
    super().__init__(lmbda)
    self.ratio_postprocess_fn = ratio_postprocess_fn
    self.conjugate_fn = conjugate_fn

  def directional(
      self, k_xx, k_yy, k_xy
  ):
    """Estimates D_f(P_y || P_x) = E_y[g(r(y))] - E_x[f*(g(r(x)))], r = dP_y/dP_x.

    Args:
      k_xx: Kernel matrix of the denominator distribution x, shape `[m, m]`.
      k_yy: Kernel matrix of the numerator distribution y, shape `[n, n]`.
      k_xy: Cross-kernel matrix between x and y, shape `[m, n]`.

    Returns:
      Scalar directional f-divergence estimate `D_f(P_y || P_x)`.
    """
    cho_x = regularized_cho_factor(k_xx, self.lmbda)
    r_x = self.likelihood_ratio_yx_on_z(k_xx, k_xy, k_xx, cho_x, k_xy)
    r_y = self.likelihood_ratio_yx_on_z(k_xy.T, k_yy, k_xx, cho_x, k_xy)
    g = self.ratio_postprocess_fn
    return jnp.mean(g(r_y)) - jnp.mean(self.conjugate_fn(g(r_x)))

  def __call__(
      self, k_xx, k_yy, k_xy
  ):
    """Estimates f-divergence from variational form and likelihood ratio.

    Args:
      k_xx: Kernel matrix of the first distribution, shape `[m, m]`.
      k_yy: Kernel matrix of the second distribution, shape `[n, n]`.
      k_xy: Kernel matrix of the cross distributions, shape `[m, n]`.

    Returns:
      The f-divergence between the two distributions (directional if
      `self.symmetric` is False, or the maximum of both directions if True).
    """
    d = self.directional(k_xx, k_yy, k_xy)
    if not self.symmetric:
      return d
    return jnp.maximum(d, self.directional(k_yy, k_xx, k_xy.T))


class HockeyStickKernelFDivergence(KernelFDivergence):
  """Hockey-stick divergence from variational form and likelihood ratio."""

  def __init__(self, lmbda, order = 1.0):
    """Initializes the instance.

    Args:
      lmbda: Positive regularization parameter.
      order: Order `e^eps` of the hockey-stick divergence (default 1.0, which
        corresponds to total variation).
    """
    g = lambda r: (r > order).astype(r.dtype)
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


def symmetric_estimator_factory(class_):
  """Returns a symmetric subclass of `class_` with `symmetric = True`.

  Args:
    class_: A `KernelFDivergence` subclass.

  Returns:
    A subclass of `class_` with `symmetric = True` and `__name__` set to
    `f"Symmetric{class_.__name__}"`.

  Example:
    ```python
    SymmetricKL = symmetric_estimator_factory(KullbackLeiblerFDivergence)
    estimator = SymmetricKL(lmbda=0.1)
    ```
  """

  class SymmetricEstimator(class_):
    symmetric = True

  SymmetricEstimator.__name__ = f"Symmetric{class_.__name__}"
  SymmetricEstimator.__qualname__ = f"Symmetric{class_.__name__}"
  return SymmetricEstimator


SymmetricHockeyStick = symmetric_estimator_factory(HockeyStickKernelFDivergence)
SymmetricKLDivergence = symmetric_estimator_factory(KullbackLeiblerFDivergence)
SymmetricTotalVariation = symmetric_estimator_factory(TotalVariationFDivergence)
SymmetricPearsonChiSquared = symmetric_estimator_factory(
    PearsonChiSquaredFDivergence
)
