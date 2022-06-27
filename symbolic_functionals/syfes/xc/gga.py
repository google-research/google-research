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

"""Existing functional forms within Generalized Gradient Approximation (GGA).

Nonmenclature: e_x and e_c denotes exchange and correlation energy density
per unit volume. Note that some codes (e.g. libxc) compute energy densities
per particle (commonly denoted by eps_x and eps_c), which are equal to e_x
and e_c divided by density (rho). Functions names containing 'unpolarized'
and 'polarized' are functions for spin unpolarized and spin polarized
molecules, respectively. In spin polarized case, variable names ending with a
and b denote spin up and spin down quantities, respectively.
"""

import jax.numpy as jnp
import numpy as onp
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import rsh
from symbolic_functionals.syfes.xc import utils


def get_reduced_density_gradient(rho, sigma, use_jax=True):
  """Evaluates reduced density gradient.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), the reduced density gradient.
  """
  np = jnp if use_jax else onp
  # NOTE(htm): EPSILON is inserted in such a way that the reduce density
  # gradient is zero if rho or sigma or both are zero.
  return np.sqrt(sigma + utils.EPSILON ** 3) / (rho ** (4 / 3) + utils.EPSILON)


def eval_e_x_using_f_x(f_x):
  """Decorator function for computing e_x for spin unpolarized case.

  Args:
    f_x: Function, evaluates exchange enhancement factor.

  Returns:
    Function, evaluates exchange energy density.
  """

  def e_x_unpolarized(rho, sigma, **kwargs):
    """Evaluates exchange energy density for spin unpolarized case.

    Args:
      rho: Float numpy array with shape (num_grids,), the electron density.
      sigma: Float numpy array with shape (num_grids,), the norm square of
        density gradient.
      **kwargs: Dict, extra arguments for f_x.

    Returns:
      Float numpy array with shape (num_grids,), the exchange energy density.
    """
    x = get_reduced_density_gradient(rho, sigma)
    return lda.e_x_lda_unpolarized(rho) * f_x(x, **kwargs)

  return e_x_unpolarized


def eval_e_x_polarized_using_e_x_unpolarized(e_x_unpolarized):
  """Decorator function for computing e_x for spin polarized case.

  The spin-scaling relation for exchange energy is used:
  e_x(rhoa, rhob) = 0.5 * (e_x(2*rhoa) + e_x(2*rhob))

  Args:
    e_x_unpolarized: function to evaluate exchange energy density
      for spin unpolarized case.

  Returns:
    Function to evaluate exchange energy density for spin polarized case.
  """

  def e_x_polarized(rhoa, rhob, sigma_aa, sigma_ab, sigma_bb, **kwargs):
    """Evaluates exchange energy density for spin polarized case.

    Args:
      rhoa: Float numpy array with shape (num_grids,), the spin up
        electron density.
      rhob: Float numpy array with shape (num_grids,), the spin down
        electron density.
      sigma_aa: Float numpy array with shape (num_grids,), the norm square
        of density gradient (aa component).
      sigma_ab: Float numpy array with shape (num_grids,), the norm square
        of density gradient (ab component).
      sigma_bb: Float numpy array with shape (num_grids,), the norm square
        of density gradient (bb component).
      **kwargs: Dict, extra arguments for e_x_unpolarized function.

    Returns:
      Float numpy array with shape (num_grids,), the exchange energy density.
    """
    del sigma_ab
    return 0.5 * (e_x_unpolarized(2 * rhoa, 4 * sigma_aa, **kwargs)
                  + e_x_unpolarized(2 * rhob, 4 * sigma_bb, **kwargs))

  return e_x_polarized


def f_x_pbe(x, kappa=0.804, mu=0.2195149727645171):
  """Evaluates PBE exchange enhancement factor.

  10.1103/PhysRevLett.77.3865 Eq. 14.
  F_X(x) = 1 + kappa ( 1 - 1 / (1 + mu s^2)/kappa )
  kappa, mu = 0.804, 0.2195149727645171 (PBE values)
  s = c x, c = 1 / (2 (3pi^2)^(1/3) )

  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    kappa: Float, parameter.
    mu: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), the PBE exchange enhancement
      factor.
  """
  c = 1 / (2 * (3 * jnp.pi ** 2) ** (1 / 3))
  s = c * x
  f_x = 1 + kappa - kappa / (1 + mu * s ** 2 / kappa)
  return f_x


e_x_pbe_unpolarized = eval_e_x_using_f_x(f_x_pbe)
e_x_pbe_polarized = eval_e_x_polarized_using_e_x_unpolarized(
    e_x_pbe_unpolarized)


def e_c_pbe_unpolarized(rho,
                        sigma,
                        beta=0.06672455060314922,
                        gamma=0.0310906908696549,
                        use_jax=True):
  """Evaluates PBE correlation energy density for spin unpolarized case.

  10.1103/PhysRevLett.77.3865 Eq. 7-8.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    beta: Float, parameter.
    gamma: Float, parameter.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), PBE correlation energy density.
  """
  np = jnp if use_jax else onp
  e_c_lda = lda.e_c_lda_unpolarized(rho, use_pbe_params=True)
  eps_c_lda = e_c_lda / (rho + utils.EPSILON)

  # t, a and h denotes quantities t, A, and H in the PBE paper.
  t = (np.pi ** (1 / 2) / (4 * (3 * np.pi ** 2) ** (1 / 6))
       * np.sqrt(sigma) / rho ** (7 / 6))
  a = beta / (gamma * (np.exp(- eps_c_lda / gamma) - 1))
  at2 = a * t ** 2
  h = gamma * np.log(
      1 + beta / gamma * t ** 2 * (1 + at2) / (1 + at2 + at2 ** 2))

  return e_c_lda + rho * h


def e_c_pbe_polarized(rhoa,
                      rhob,
                      sigma_aa,
                      sigma_ab,
                      sigma_bb,
                      beta=0.06672455060314922,
                      gamma=0.0310906908696549):
  """Evaluates PBE correlation energy density for spin polarized case.

  10.1103/PhysRevLett.77.3865 Eq. 7-8.

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.
    sigma_aa: Float numpy array with shape (num_grids,), the norm square
      of density gradient (aa component).
    sigma_ab: Float numpy array with shape (num_grids,), the norm square
      of density gradient (ab component).
    sigma_bb: Float numpy array with shape (num_grids,), the norm square
      of density gradient (bb component).
    beta: Float, parameter.
    gamma: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), PBE correlation energy density.
  """
  rho = rhoa + rhob
  zeta = (rhoa - rhob) / (rho + utils.EPSILON)
  rhograd = jnp.sqrt(sigma_aa + 2 * sigma_ab + sigma_bb)
  e_c_lda = lda.e_c_lda_polarized(rhoa, rhob, use_pbe_params=True)
  eps_c_lda = e_c_lda / (rho + utils.EPSILON)

  # phi, t, a and h denotes quantities \phi, t, A, and H in the PBE paper.
  phi = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
  t = (jnp.pi ** (1 / 2) / (4 * (3 * jnp.pi ** 2) ** (1 / 6) * phi)
       * rhograd / rho ** (7 / 6))
  a = beta / (gamma * (jnp.exp(- eps_c_lda / (gamma * phi ** 3)) - 1))
  at2 = a * t ** 2
  h = gamma * phi ** 3 * jnp.log(
      1 + beta / gamma * t ** 2 * (1 + at2) / (1 + at2 + at2 ** 2))
  return e_c_lda + rho * h


def f_x_rpbe(x, kappa=0.804, mu=0.2195149727645171, use_jax=True):
  """Evaluates RPBE exchange enhancement factor.

  10.1103/PhysRevB.59.7413 Eq. 15
  F_X(x) = 1 + kappa ( 1 - e^( - mu s^2 / kappa ) )
  kappa, mu = 0.804, 0.21951 (PBE/RPBE values)
  s = c x, c = 1 / (2 (3pi^2)^(1/3) )

  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    kappa: Float, parameter.
    mu: Float, parameter.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), F_X.
  """
  np = jnp if use_jax else onp
  c = 1 / (2 * (3 * np.pi ** 2) ** (1 / 3))
  s = c * x
  return 1 + kappa * (1 - np.exp(- mu * s ** 2 / kappa))


e_x_rpbe_unpolarized = eval_e_x_using_f_x(f_x_rpbe)
e_x_rpbe_polarized = eval_e_x_polarized_using_e_x_unpolarized(
    e_x_rpbe_unpolarized)


def f_x_b88(x, beta=0.0042, use_jax=True):
  """Evaluates B88 exchange enhancement factor.

  10.1103/PhysRevA.38.3098 Eq. 8 converted to spin restricted form
  F_X = 1 + c1 x^2 / (1 + c2 * x * arcsinh [2^(1/3) x])
  c1 = 8 * (1/2)^(2/3) * (pi/3)*(1/3) / 3 * beta
  c2 = 6 * 2^(1/3) * beta

  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    beta: Float, parameter.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), F_X.
  """
  np = jnp if use_jax else onp
  c1 = 8 * (1 / 2) ** (2 / 3) * (np.pi / 3) ** (1 / 3) / 3 * beta
  c2 = 6 * 2 ** (1 / 3) * beta
  return 1 + c1 * x ** 2 / (1 + c2 * x * np.arcsinh(2 ** (1 / 3) * x))


e_x_b88_unpolarized = eval_e_x_using_f_x(f_x_b88)
e_x_b88_polarized = eval_e_x_polarized_using_e_x_unpolarized(
    e_x_b88_unpolarized)


def u_b97(x, gamma, polarized=False):
  """Evaluates auxiliary quantity u for B97-based functionals.

  10.1063/1.475007.
  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    gamma: Float, parameter.
    polarized: Boolean, whether the system is spin polarized.

  Returns:
    Float numpy array with shape (num_grids,), the auxiliary quantity u.
  """
  spin_factor = 1 if polarized else (1 / 2) ** (-2 / 3)
  s2 = spin_factor * x ** 2
  return gamma * s2 / (1 + gamma * s2)


def f_b97(x,
          power_series=((0, 0.8094), (1, 0.5073), (2, 0.7481)),
          gamma=0.004,
          polarized=False):
  """Evaluates B97 enhancement factor.

  10.1063/1.475007.
  F_X = sum_i coeffs[i] * u^i    i = 0, 1, ..., num_coeffs - 1
  u = gamma s^2 / (1 + gamma s^2), s^2 = spin_factor * x^2, where the
  spin_factor is (1/2)^(-2/3) and 1 for spin unpolarized and polarized case.
  The default values of the function arguments correspond to B97 exchange.

  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    power_series: List of tuples of (integer, float), defines the power
      series expansion of u. Each tuple corresponds to a term with given
      exponent (first element) and a linear coefficient (second element).
    gamma: Float, parameter.
    polarized: Boolean, whether the system is spin polarized.

  Returns:
    Float numpy array with shape (num_grids,), F_XC.
  """
  u = u_b97(x, gamma, polarized=polarized)
  f_x = jnp.zeros_like(x)
  for exponent, coeff in power_series:
    f_x += coeff * u ** exponent
  return f_x


e_x_b97_unpolarized = eval_e_x_using_f_x(f_b97)
e_x_b97_polarized = eval_e_x_polarized_using_e_x_unpolarized(
    e_x_b97_unpolarized)


def e_c_b97_unpolarized(rho,
                        sigma,
                        power_series_ss=(
                            (0, 0.1737), (1, 2.3487), (2, -2.4868)
                            ),
                        power_series_os=(
                            (0, 0.9454), (1, 0.7471), (2, -4.5961)
                            ),
                        gamma_ss=0.2,
                        gamma_os=0.006):
  """Evaluates B97 correlation energy density for spin unpolarized case.

  10.1063/1.475007 Eq. 4-6.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    power_series_ss: List of tuples of (integer, float), defines the power
      series expansion of u for same-spin correlation.
    power_series_os: List of tuples of (integer, float), defines the power
      series expansion of u for opposite-spin correlation.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), B97 correlation energy density.
  """
  e_c_lda_ss, e_c_lda_os = lda.decomposed_e_c_lda_unpolarized(rho)

  x = get_reduced_density_gradient(rho, sigma)
  f_c_ss = f_b97(
      x, power_series=power_series_ss, gamma=gamma_ss, polarized=False)
  f_c_os = f_b97(
      x, power_series=power_series_os, gamma=gamma_os, polarized=False)

  return e_c_lda_ss * f_c_ss + e_c_lda_os * f_c_os


def e_c_b97_polarized(rhoa,
                      rhob,
                      sigma_aa,
                      sigma_ab,
                      sigma_bb,
                      power_series_ss=(
                          (0, 0.1737), (1, 2.3487), (2, -2.4868)
                          ),
                      power_series_os=(
                          (0, 0.9454), (1, 0.7471), (2, -4.5961)
                          ),
                      gamma_ss=0.2,
                      gamma_os=0.006):
  """Evaluates B97 correlation energy density for spin polarized case.

  10.1063/1.475007 Eq. 4-6.

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.
    sigma_aa: Float numpy array with shape (num_grids,), the norm square of
      density gradient (aa component).
    sigma_ab: Float numpy array with shape (num_grids,), the norm square of
      density gradient (ab component).
    sigma_bb: Float numpy array with shape (num_grids,), the norm square of
      density gradient (bb component).
    power_series_ss: List of tuples of (integer, float), defines the power
      series expansion of u for same-spin correlation.
    power_series_os: List of tuples of (integer, float), defines the power
      series expansion of u for opposite-spin correlation.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), B97 correlation energy density.
  """
  del sigma_ab

  e_c_lda_aa, e_c_lda_bb, e_c_lda_ab = lda.decomposed_e_c_lda_polarized(
      rhoa, rhob)

  xa = get_reduced_density_gradient(rhoa, sigma_aa)
  xb = get_reduced_density_gradient(rhob, sigma_bb)
  xave = jnp.sqrt(0.5 * (xa ** 2 + xb ** 2))

  f_c_aa = f_b97(
      xa, power_series=power_series_ss, gamma=gamma_ss, polarized=True)
  f_c_bb = f_b97(
      xb, power_series=power_series_ss, gamma=gamma_ss, polarized=True)
  f_c_ab = f_b97(
      xave, power_series=power_series_os, gamma=gamma_os, polarized=True)

  return e_c_lda_aa * f_c_aa + e_c_lda_bb * f_c_bb + e_c_lda_ab * f_c_ab


def e_x_wb97_unpolarized(rho,
                         sigma,
                         power_series,
                         gamma,
                         omega):
  """Evaluates B97 short range exchange energy density for spin unpolarized case.

  10.1039/c3cp54374a

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    power_series: List of tuples of (integer, float), defines the power
      series expansion of u for exchange.
    gamma: Float, parameter.
    omega: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), wB97X-V xc energy density.
  """
  return rsh.f_rsh(rho, omega=omega, polarized=False) * e_x_b97_unpolarized(
      rho,
      sigma,
      power_series=power_series,
      gamma=gamma)


e_x_wb97_polarized = eval_e_x_polarized_using_e_x_unpolarized(
    e_x_wb97_unpolarized)


def e_xc_wb97xv_unpolarized(rho,
                            sigma,
                            power_series_x=(
                                (0, 0.833), (1, 0.603), (2, 1.194),),
                            power_series_ss=((0, 0.556), (1, -0.257),),
                            power_series_os=((0, 1.219), (1, -1.850),),
                            gamma_x=0.004,
                            gamma_ss=0.2,
                            gamma_os=0.006,
                            omega=0.3):
  """Evaluates wB97X-V xc energy density for spin unpolarized case.

  10.1039/c3cp54374a

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    power_series_x: List of tuples of (integer, float), defines the power
      series expansion of u for exchange.
    power_series_ss: List of tuples of (integer, float), defines the power
      series expansion of u for same-spin correlation.
    power_series_os: List of tuples of (integer, float), defines the power
      series expansion of u for opposite-spin correlation.
    gamma_x: Float, parameter.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.
    omega: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), wB97X-V xc energy density.
  """
  e_x = e_x_wb97_unpolarized(
      rho,
      sigma,
      power_series=power_series_x,
      gamma=gamma_x,
      omega=omega)
  e_c = e_c_b97_unpolarized(
      rho,
      sigma,
      power_series_ss=power_series_ss,
      power_series_os=power_series_os,
      gamma_ss=gamma_ss,
      gamma_os=gamma_os)
  return e_x + e_c


def e_xc_wb97xv_polarized(rhoa,
                          rhob,
                          sigma_aa,
                          sigma_ab,
                          sigma_bb,
                          power_series_x=((0, 0.833), (1, 0.603), (2, 1.194),),
                          power_series_ss=((0, 0.556), (1, -0.257),),
                          power_series_os=((0, 1.219), (1, -1.850),),
                          gamma_x=0.004,
                          gamma_ss=0.2,
                          gamma_os=0.006,
                          omega=0.3):
  """Evaluates wB97X-V xc energy density for spin unpolarized case.

  10.1039/c3cp54374a

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.
    sigma_aa: Float numpy array with shape (num_grids,), the norm square of
      density gradient (aa component).
    sigma_ab: Float numpy array with shape (num_grids,), the norm square of
      density gradient (ab component).
    sigma_bb: Float numpy array with shape (num_grids,), the norm square of
      density gradient (bb component).
    power_series_x: List of tuples of (integer, float), defines the power
      series expansion of u for exchange.
    power_series_ss: List of tuples of (integer, float), defines the power
      series expansion of u for same-spin correlation.
    power_series_os: List of tuples of (integer, float), defines the power
      series expansion of u for opposite-spin correlation.
    gamma_x: Float, parameter.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.
    omega: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), wB97X-V xc energy density.
  """
  e_x = e_x_wb97_polarized(
      rhoa,
      rhob,
      sigma_aa,
      sigma_ab,
      sigma_bb,
      power_series=power_series_x,
      gamma=gamma_x,
      omega=omega)
  e_c = e_c_b97_polarized(
      rhoa,
      rhob,
      sigma_aa,
      sigma_ab,
      sigma_bb,
      power_series_ss=power_series_ss,
      power_series_os=power_series_os,
      gamma_ss=gamma_ss,
      gamma_os=gamma_os)
  return e_x + e_c
