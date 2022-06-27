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

"""Existing functional forms within Local Denisty Approximation (LDA).

Nonmenclature: e_x and e_c denotes exchange and correlation energy density
per unit volume. Note that some codes (e.g. libxc) compute energy densities
per particle (commonly denoted by eps_x and eps_c), which are equal to e_x
and e_c divided by density (rho). Functions names containing 'unpolarized'
and 'polarized' are functions for spin unpolarized and spin polarized
molecules, respectively. In spin polarized case, variable names ending with a
and b denote spin up and spin down quantities, respectively.
"""

import copy
import jax.numpy as jnp
import numpy as onp
from symbolic_functionals.syfes.xc import utils


# _LDA_PW_PARAMETERS and _LDA_PW_PBE_PARAMETERS: dictionaries containing
# parameters a, alpha, beta1, beta2, beta3, beta4 and fpp_0, which are used to
# compute LDA correlation functional in the PW parametrization. The first six
# parameters are used to evaluate auxiliary G functions in the PW paper
# (10.1103/PhysRevB.45.13244) and their values in _LDA_PW_PARAMETERS values are
# taken from Table I of the PW paper. fpp_0 denotes the second derivative of
# f_zeta function in the PW paper at zeta = 0. PBE correlation functional uses
# PW parametrization of LDA correlation with slightly different number of digits
# for some parameters, as defined in _LDA_PW_PBE_PARAMETERS.

_LDA_PW_PARAMETERS = {
    'eps_c_unpolarized': {
        'a': 0.031091,
        'alpha1': 0.21370,
        'beta1': 7.5957,
        'beta2': 3.5876,
        'beta3': 1.6382,
        'beta4': 0.49294},
    'eps_c_polarized': {
        'a': 0.015545,
        'alpha1': 0.20548,
        'beta1': 14.1189,
        'beta2': 6.1977,
        'beta3': 3.3662,
        'beta4': 0.62517},
    'alpha_c': {
        'a': 0.016887,
        'alpha1': 0.11125,
        'beta1': 10.357,
        'beta2': 3.6231,
        'beta3': 0.88026,
        'beta4': 0.49671},
    'fpp_0': 1.709921
}

_LDA_PW_PBE_PARAMETERS = copy.deepcopy(_LDA_PW_PARAMETERS)
_LDA_PW_PBE_PARAMETERS['eps_c_unpolarized'].update({'a': 0.0310907})
_LDA_PW_PBE_PARAMETERS['eps_c_polarized'].update({'a': 0.01554535})
_LDA_PW_PBE_PARAMETERS['alpha_c'].update({'a': 0.0168869})
_LDA_PW_PBE_PARAMETERS['fpp_0'] = 1.709920934161365617563962776245


def e_x_lda_unpolarized(rho):
  """Evaluates LDA exchange energy density for spin unpolarized case.

  Parr & Yang Density-functional theory of atoms and molecules Eq. 6.1.20.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.

  Returns:
    Float numpy array with shape (num_grids,), the LDA exchange energy density.
  """
  return -3 / 4 * (3 / jnp.pi) ** (1 / 3) * rho ** (4 / 3)


def e_x_lda_polarized(rhoa, rhob):
  """Evaluates LDA exchange energy density for spin polarized case.

  Parr & Yang Density-functional theory of atoms and molecules Eq. 8.2.18.

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.

  Returns:
    Float numpy array with shape (num_grids,), the LDA exchange energy density.
  """
  rho = rhoa + rhob
  zeta = (rhoa - rhob) / (rho + utils.EPSILON)  # spin polarization
  spin_scaling = 0.5 * ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3))
  return - 3 / 4 * (3 / jnp.pi) ** (1 / 3) * rho ** (4 / 3) * spin_scaling


def g_lda_pw(rs, a, alpha1, beta1, beta2, beta3, beta4, use_jax=True):
  """Evaluates auxiliary function G in the PW parametrization of LDA functional.

  10.1103/PhysRevB.45.13244 Eq. 10.

  Args:
    rs: Float numpy array with shape (num_grids,), Wigner-Seitz radius.
    a: Float, parameter.
    alpha1: Float, parameter.
    beta1: Float, parameter.
    beta2: Float, parameter.
    beta3: Float, parameter.
    beta4: Float, parameter.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), auxiliary function G.
  """
  np = jnp if use_jax else onp
  den = 2 * a * (
      beta1 * rs**(1 / 2) + beta2 * rs + beta3 * rs**(3 / 2) + beta4 * rs**2)
  return -2 * a * (1 + alpha1 * rs) * np.log(1 + 1 / den)


def get_wigner_seitz_radius(rho):
  """Evaluates Wigner-Seitz radius of given density.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.

  Returns:
    Float numpy array with shape (num_grids,), the Wigner-Seitz radius.
  """
  return (3 / (4 * jnp.pi)) ** (1/3) * (rho + utils.EPSILON) ** (-1 / 3)


def e_c_lda_unpolarized(rho, use_pbe_params=False, use_jax=True):
  """Evaluates LDA correlation energy density for spin unpolarized case.

  PW parametrization. 10.1103/PhysRevB.45.13244 Eq. 8-9.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    use_pbe_params: Boolean, whether use PBE parameters.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), the LDA correlation energy
      density.
  """
  rs = get_wigner_seitz_radius(rho)

  if use_pbe_params:
    g_params = _LDA_PW_PBE_PARAMETERS
  else:
    g_params = _LDA_PW_PARAMETERS

  return rho * g_lda_pw(rs=rs, **g_params['eps_c_unpolarized'], use_jax=use_jax)


def e_c_lda_polarized(rhoa, rhob, use_pbe_params=False, use_jax=True):
  """Evaluates LDA correlation energy density for spin polarized case.

  PW parametrization. 10.1103/PhysRevB.45.13244 Eq. 8-9.

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.
    use_pbe_params: Boolean, whether use PBE parameters.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), the LDA correlation energy
      density.
  """
  rho = rhoa + rhob
  rs = get_wigner_seitz_radius(rho)
  zeta = (rhoa - rhob) / (rho + utils.EPSILON)

  # spin dependent interpolation coefficient
  f_zeta = 1 / (2 ** (4 / 3) - 2) * (
      (1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2)

  if use_pbe_params:
    params = _LDA_PW_PBE_PARAMETERS
  else:
    params = _LDA_PW_PARAMETERS

  eps_c_unpolarized = g_lda_pw(
      rs=rs, **params['eps_c_unpolarized'], use_jax=use_jax)
  eps_c_polarized = g_lda_pw(
      rs=rs, **params['eps_c_polarized'], use_jax=use_jax)
  alpha_c = -g_lda_pw(rs=rs, **params['alpha_c'], use_jax=use_jax)

  return rho * (
      eps_c_unpolarized
      + (1 / params['fpp_0']) * alpha_c * f_zeta * (1 - zeta ** 4)
      + (eps_c_polarized - eps_c_unpolarized) * f_zeta * zeta ** 4
      )


def decomposed_e_c_lda_unpolarized(rho, use_jax=True):
  """Evaluates LDA e_c decomposed into same-spin and opposite-spin components.

  This function returns the LDA correlation energy density partitioned into
  same-spin and opposite-spin components. 10.1063/1.475007 Eq. 7-8.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    e_c_ss: Float numpy array with shape (num_grids,), the same-spin
      component of LDA correlation energy density.
    e_c_os: Float numpy array with shape (num_grids,), the opposite-spin
      component of LDA correlation energy density.
  """
  np = jnp if use_jax else onp
  e_c = e_c_lda_unpolarized(rho, use_jax=use_jax)
  e_c_ss = 2 * e_c_lda_polarized(rho / 2, np.zeros_like(rho), use_jax=use_jax)
  e_c_os = e_c - e_c_ss

  return e_c_ss, e_c_os


def decomposed_e_c_lda_polarized(rhoa, rhob, use_jax=True):
  """Evaluates LDA e_c decomposed into same-spin and opposite-spin components.

  This function returns the LDA correlation energy density partitioned into
  same-spin and opposite-spin components. 10.1063/1.475007 Eq. 7-8.

  Args:
    rhoa: Float numpy array with shape (num_grids,), the spin up
      electron density.
    rhob: Float numpy array with shape (num_grids,), the spin down
      electron density.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    e_c_aa: Float numpy array with shape (num_grids,), the same-spin (aa)
      component of LDA correlation energy density.
    e_c_bb: Float numpy array with shape (num_grids,), the same-spin (bb)
      component of LDA correlation energy density.
    e_c_ab: Float numpy array with shape (num_grids,), the opposite-spin
      component of LDA correlation energy density.
  """
  np = jnp if use_jax else onp
  zero = np.zeros_like(rhoa)

  e_c = e_c_lda_polarized(rhoa, rhob, use_jax=use_jax)
  e_c_aa = e_c_lda_polarized(rhoa, zero, use_jax=use_jax)
  e_c_bb = e_c_lda_polarized(zero, rhob, use_jax=use_jax)
  e_c_ab = e_c - e_c_aa - e_c_bb

  return e_c_aa, e_c_bb, e_c_ab
