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

"""Existing meta-GGA functional forms.

Nonmenclature: e_x and e_c denotes exchange and correlation energy density
per unit volume. Note that some codes (e.g. libxc) compute energy densities
per particle (commonly denoted by eps_x and eps_c), which are equal to e_x
and e_c divided by density (rho). Functions names containing 'unpolarized'
and 'polarized' are functions for spin unpolarized and spin polarized
molecules, respectively. In spin polarized case, variable names ending with a
and b denote spin up and spin down quantities, respectively.
"""

import functools
import jax.numpy as jnp
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import rsh
from symbolic_functionals.syfes.xc import utils


# WB97MV_PARAMS and B97MV_PARAMS can both be used as input
# parameters for e_xc_wb97mv_unpolarized/e_xc_wb97mv_polarized functions, which
# correspond to wB97M-V and B97M-V functionals, respectively. These parametesr
# are consistent with libxc/src/mgga_xc_b97mv.c

WB97MV_PARAMS = {
    'power_series_x': ((0, 0, 0.85), (1, 0, 0.259), (0, 1, 1.007)),
    'power_series_ss': ((0, 0, 0.443), (1, 0, -4.535), (2, 0, -3.39),
                        (4, 3, 4.278), (0, 4, -1.437)),
    'power_series_os': ((0, 0, 1.000), (1, 0, 1.358), (2, 0, 2.924),
                        (6, 0, -1.39), (2, 1, -8.812), (6, 1, 9.142)),
    'gamma_x': 0.004,
    'gamma_ss': 0.2,
    'gamma_os': 0.006,
    'omega': 0.3
}


B97MV_PARAMS = {
    'power_series_x': ((0, 0, 1.000), (0, 1, 1.308), (0, 2, 1.901),
                       (1, 0, 0.416), (1, 1, 3.070)),
    'power_series_ss': ((0, 0, 1.000), (0, 2, -1.855), (1, 0, -5.668),
                        (3, 2, -20.497), (4, 2, -20.364)),
    'power_series_os': ((0, 0, 1.000), (0, 1, 1.573), (0, 3, -6.298),
                        (1, 0, 2.535), (3, 2, -6.427)),
    'gamma_x': 0.004,
    'gamma_ss': 0.2,
    'gamma_os': 0.006,
    'omega': 0.0
}


def get_mgga_t(rho, tau, polarized):
  """Evaluates the auxiliary quantity t in meta-GGA functional forms.

  t = (tau_HEG / tau),  where tau_HEG is the kinetic energy density of
  homogeneous electron gass.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    tau: Float numpy array with shape (num_grids,), the kinetic energy density.
    polarized: Boolean, whether the system is spin polarized.

  Returns:
    Float numpy array with shape (num_grids,), the auxiliary quantity w.
  """
  spin_factor = 1 if polarized else (1 / 2) ** (2 / 3)
  # 3 / 10 instead of 3 / 5 is used below because tau is defined with 1 / 2
  tau_heg = 3 / 10 * (6 * jnp.pi ** 2) ** (2 / 3) * rho ** (5 / 3)
  return spin_factor * tau_heg / (tau + utils.EPSILON)


def f_b97m(x, t, power_series, gamma, polarized):
  """Evaluates wB97M enhancement factor.

  10.1063/1.4952647 Eq. 13.
  f = sum_i,j coeffs[i,j] * w^i * u^j
  w, u are auxiliary quantities evaluated by w_b97m and gga.u_b97, respectively.
  In 10.1063/1.4952647 the enhancement factor is denoted as g, here we use f
  to be consistent enhancement factors in gga.py. Also note that the reduced
  density gradient x is denoted as s in many B97-based papers.

  Args:
    x: Float numpy array with shape (num_grids,), the reduced density gradient.
    t: Float numpy array with shape (num_grids,), an auxiliary quantity related
      to kinetic energy density.
    power_series: List of tuples of (integer, integer, float), defines the power
      series expansion of w and u. Each tuple contains the exponent of w, u and
      the prefactor.
    gamma: Float, parameter.
    polarized: Boolean, whether the system is spin polarized.

  Returns:
    Float numpy array with shape (num_grids,), the wB97M enhancement factor.
  """
  u = gga.u_b97(x, gamma, polarized=polarized)
  w = (t - 1) / (t + 1)
  f_x = jnp.zeros_like(x)
  for w_exponent, u_exponent, coeff in power_series:
    f_x += coeff * w ** w_exponent * u ** u_exponent
  return f_x


def e_xc_wb97mv_unpolarized(rho,
                            sigma,
                            tau,
                            power_series_x=WB97MV_PARAMS['power_series_x'],
                            power_series_ss=WB97MV_PARAMS['power_series_ss'],
                            power_series_os=WB97MV_PARAMS['power_series_os'],
                            gamma_x=WB97MV_PARAMS['gamma_x'],
                            gamma_ss=WB97MV_PARAMS['gamma_ss'],
                            gamma_os=WB97MV_PARAMS['gamma_os'],
                            omega=WB97MV_PARAMS['omega']):
  """Evaluates wB97M-V xc energy density for spin unpolarized case.

  10.1063/1.4952647

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    sigma: Float numpy array with shape (num_grids,), the norm square of
      density gradient.
    tau: Float numpy array with shape (num_grids,), the kinetic energy density.
    power_series_x: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for exchange.
    power_series_ss: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for same-spin correlation.
    power_series_os: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for opposite-spin correlation.
    gamma_x: Float, parameter.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.
    omega: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), wB97M-V xc energy density.
  """
  x = gga.get_reduced_density_gradient(rho, sigma)
  t = get_mgga_t(rho, tau, polarized=False)

  e_x_lda = lda.e_x_lda_unpolarized(rho)
  e_c_lda_ss, e_c_lda_os = lda.decomposed_e_c_lda_unpolarized(rho)

  f_x = f_b97m(
      x,
      t,
      power_series=power_series_x,
      gamma=gamma_x,
      polarized=False)
  f_c_ss = f_b97m(
      x,
      t,
      power_series=power_series_ss,
      gamma=gamma_ss,
      polarized=False)
  f_c_os = f_b97m(
      x,
      t,
      power_series=power_series_os,
      gamma=gamma_os,
      polarized=False)

  return (rsh.f_rsh(rho, omega=omega, polarized=False) * e_x_lda * f_x
          + e_c_lda_ss * f_c_ss + e_c_lda_os * f_c_os)


def e_xc_wb97mv_polarized(rhoa,
                          rhob,
                          sigma_aa,
                          sigma_ab,
                          sigma_bb,
                          tau_a,
                          tau_b,
                          power_series_x=WB97MV_PARAMS['power_series_x'],
                          power_series_ss=WB97MV_PARAMS['power_series_ss'],
                          power_series_os=WB97MV_PARAMS['power_series_os'],
                          gamma_x=WB97MV_PARAMS['gamma_x'],
                          gamma_ss=WB97MV_PARAMS['gamma_ss'],
                          gamma_os=WB97MV_PARAMS['gamma_os'],
                          omega=WB97MV_PARAMS['omega']):
  """Evaluates wB97M-V xc energy density for spin polarized case.

  10.1063/1.4952647

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up electron
      density.
    rhob: Float numpy array with shape (num_grids,), the spin down electron
      density.
    sigma_aa: Float numpy array with shape (num_grids,), the norm square of
      density gradient (aa component).
    sigma_ab: Float numpy array with shape (num_grids,), the norm square of
      density gradient (ab component).
    sigma_bb: Float numpy array with shape (num_grids,), the norm square of
      density gradient (bb component).
    tau_a: Float numpy array with shape (num_grids,), the spin up kinetic
      energy density.
    tau_b: Float numpy array with shape (num_grids,), the spin down kinetic
      energy density.
    power_series_x: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for exchange.
    power_series_ss: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for same-spin correlation.
    power_series_os: List of tuples of (integer, integer, float), defines the
      power series expansion of w and u for opposite-spin correlation.
    gamma_x: Float, parameter.
    gamma_ss: Float, parameter.
    gamma_os: Float, parameter.
    omega: Float, parameter.

  Returns:
    Float numpy array with shape (num_grids,), wB97M-V xc energy density.
  """
  del sigma_ab

  xa = gga.get_reduced_density_gradient(rhoa, sigma_aa)
  xb = gga.get_reduced_density_gradient(rhob, sigma_bb)
  xave = jnp.sqrt(0.5 * (xa ** 2 + xb ** 2))

  ta = get_mgga_t(rhoa, tau_a, polarized=True)
  tb = get_mgga_t(rhob, tau_b, polarized=True)
  tave = 0.5 * (ta + tb)

  # use e_x_lda_unpolarized and spin scaling to compute e_x_lda_a and e_x_lda_b
  # separately
  e_x_lda_a = 0.5 * lda.e_x_lda_unpolarized(2 * rhoa)
  e_x_lda_b = 0.5 * lda.e_x_lda_unpolarized(2 * rhob)
  e_c_lda_aa, e_c_lda_bb, e_c_lda_ab = lda.decomposed_e_c_lda_polarized(
      rhoa, rhob)

  f_x_a = f_b97m(
      xa,
      ta,
      power_series=power_series_x,
      gamma=gamma_x,
      polarized=True)
  f_x_b = f_b97m(
      xb,
      tb,
      power_series=power_series_x,
      gamma=gamma_x,
      polarized=True)
  f_c_aa = f_b97m(
      xa,
      ta,
      power_series=power_series_ss,
      gamma=gamma_ss,
      polarized=True)
  f_c_bb = f_b97m(
      xb,
      tb,
      power_series=power_series_ss,
      gamma=gamma_ss,
      polarized=True)
  f_c_ab = f_b97m(
      xave,
      tave,
      power_series=power_series_os,
      gamma=gamma_os,
      polarized=True)

  return (rsh.f_rsh(rhoa, omega=omega, polarized=True) * e_x_lda_a * f_x_a
          + rsh.f_rsh(rhob, omega=omega, polarized=True) * e_x_lda_b * f_x_b
          + e_c_lda_aa * f_c_aa + e_c_lda_bb * f_c_bb + e_c_lda_ab * f_c_ab)


e_xc_b97mv_unpolarized = functools.partial(
    e_xc_wb97mv_unpolarized, **B97MV_PARAMS)


e_xc_b97mv_polarized = functools.partial(
    e_xc_wb97mv_polarized, **B97MV_PARAMS)
